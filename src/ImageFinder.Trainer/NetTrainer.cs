using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace ImageFinder.Trainer
{
    public sealed class TrainOptions
    {
        public int Steps { get; set; } = 3000;
        public int BatchSize { get; set; } = 32;
        public float LearningRate { get; set; } = 2e-3f;
        public float WeightDecay { get; set; } = 1e-4f;
        public float PositiveFraction { get; set; } = 0.4f;
        public float HardNegativeFraction { get; set; } = 0.5f;
        public int HardMiningInterval { get; set; } = 250;
        public float PositiveWeight { get; set; } = 3f;
        public float RegressionWeight { get; set; } = 2f;
        public int Seed { get; set; } = 1;

        /// <summary>Vor dem ersten Schritt Fehlalarme suchen (sinnvoll beim Weitertrainieren).</summary>
        public bool MineAtStart { get; set; }
    }

    /// <summary>Backpropagation + Adam für das Netz aus ImageFinder.Core.</summary>
    public sealed class NetTrainer
    {
        private readonly Network _net;
        private readonly TrainOptions _opt;
        private readonly float[][] _mW, _vW, _mB, _vB;
        private int _t;

        private readonly List<(Sample Sample, float X, float Y)> _hard = new List<(Sample, float, float)>();

        public NetTrainer(Network net, TrainOptions opt)
        {
            _net = net;
            _opt = opt;
            int n = net.Layers.Count;
            _mW = new float[n][]; _vW = new float[n][]; _mB = new float[n][]; _vB = new float[n][];
            for (int i = 0; i < n; i++)
            {
                _mW[i] = new float[net.Layers[i].Weights.Length];
                _vW[i] = new float[net.Layers[i].Weights.Length];
                _mB[i] = new float[net.Layers[i].Bias.Length];
                _vB[i] = new float[net.Layers[i].Bias.Length];
            }
        }

        public void Train(Dataset data)
        {
            var rnd = new Random(_opt.Seed);
            List<Sample> positives = data.Train.Where(s => s.Objects.Count > 0).ToList();
            List<Sample> realImages = data.Train.Where(s => !s.Synthetic).ToList();
            if (positives.Count == 0) throw new InvalidOperationException("Keine Positivbeispiele.");

            var sampler = new PatchSampler();
            int B = _opt.BatchSize;
            var patches = new Patch[B];
            for (int i = 0; i < B; i++) patches[i] = new Patch(PatchSampler.Size, PatchSampler.Cells);
            var seeds = new int[B];

            var sw = Stopwatch.StartNew();
            double lossAcc = 0;
            int lossCount = 0;

            if (_opt.MineAtStart)
                MineHardNegatives(realImages);

            for (int step = 1; step <= _opt.Steps; step++)
            {
                // Batch zusammenstellen (Zufall im Haupt-Thread, damit reproduzierbar)
                var jobs = new (Sample S, (float, float)? Focus, bool Pos)[B];
                for (int b = 0; b < B; b++)
                {
                    seeds[b] = rnd.Next();
                    if (rnd.NextDouble() < _opt.PositiveFraction)
                    {
                        jobs[b] = (positives[rnd.Next(positives.Count)], null, true);
                    }
                    else if (_hard.Count > 0 && rnd.NextDouble() < _opt.HardNegativeFraction)
                    {
                        var h = _hard[rnd.Next(_hard.Count)];
                        jobs[b] = (h.Sample, (h.X, h.Y), false);
                    }
                    else
                    {
                        // echte Bilder bevorzugen, synthetische sind klein
                        Sample s = rnd.NextDouble() < 0.8 ? realImages[rnd.Next(realImages.Count)]
                                                          : data.Train[rnd.Next(data.Train.Count)];
                        jobs[b] = (s, null, false);
                    }
                }

                Gradients total = new Gradients(_net);
                object sync = new object();
                double batchLoss = 0;

                Parallel.For(0, B,
                    () => new Gradients(_net),
                    (b, _, g) =>
                    {
                        var r = new Random(seeds[b]);
                        sampler.Fill(patches[b], jobs[b].S, r, jobs[b].Focus, jobs[b].Pos);
                        g.Loss += Backprop(patches[b], g);
                        return g;
                    },
                    g =>
                    {
                        lock (sync)
                        {
                            total.Add(g);
                            batchLoss += g.Loss;
                        }
                    });

                float lr = LearningRateAt(step);
                ApplyAdam(total, lr, 1f / B);

                lossAcc += batchLoss / B;
                lossCount++;
                if (step % 50 == 0 || step == 1)
                {
                    Console.WriteLine($"Schritt {step,5}/{_opt.Steps}  Loss {lossAcc / lossCount:F4}  lr {lr:E1}  " +
                                      $"Hard-Negatives {_hard.Count}  {sw.Elapsed:mm\\:ss}");
                    lossAcc = 0;
                    lossCount = 0;
                }

                if (step % _opt.HardMiningInterval == 0 && step < _opt.Steps)
                    MineHardNegatives(realImages);
            }
        }

        private float LearningRateAt(int step)
        {
            // kurze Aufwärmphase, dann Kosinus-Abfall auf 5 %
            float warm = Math.Min(1f, step / 100f);
            double t = (double)step / _opt.Steps;
            return (float)(_opt.LearningRate * warm * (0.05 + 0.95 * 0.5 * (1 + Math.Cos(Math.PI * t))));
        }

        /// <summary>Sucht in den echten Bildern Fehlalarme und merkt sie für gezieltes Nachtraining.</summary>
        private void MineHardNegatives(List<Sample> images)
        {
            var finder = new ObjectFinder(_net) { MinDistance = 16 };
            int added = 0;
            foreach (Sample s in images)
            {
                foreach (Detection d in finder.FindAll(s.Image, 0.15f))
                {
                    bool nearObject = s.Objects.Any(o => Dist(o.X, o.Y, d.X, d.Y) < 20);
                    if (nearObject) continue;
                    _hard.Add((s, d.X, d.Y));
                    added++;
                }
            }
            const int max = 3000;
            if (_hard.Count > max) _hard.RemoveRange(0, _hard.Count - max);
            Console.WriteLine($"  Hard-Negative-Suche: {added} neue Fehlalarme gefunden");
        }

        private static float Dist(float ax, float ay, float bx, float by) =>
            MathF.Sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));

        /// <summary>Vorwärts + Rückwärts für einen Patch; addiert Gradienten in g, liefert den Loss.</summary>
        private float Backprop(Patch p, Gradients g)
        {
            int S = PatchSampler.Size;
            float[][] outs = _net.ForwardAll(p.Input, S, S, parallel: false, out (int H, int W)[] sizes);
            int last = outs.Length - 1;
            float[] o = outs[last];
            int cells = sizes[last].H * sizes[last].W;
            if (sizes[last].H != PatchSampler.Cells) throw new InvalidOperationException("Gittergröße passt nicht.");

            var grad = new float[o.Length];
            float loss = 0;
            for (int k = 0; k < cells; k++)
            {
                sbyte label = p.Label[k];
                if (label < 0) continue;
                float logit = o[k];
                float prob = ScoreMap.Sigmoid(logit);
                if (label == 1)
                {
                    float w = _opt.PositiveWeight;
                    loss += w * Softplus(-logit);
                    grad[k] = w * (prob - 1f);

                    float ex = o[cells + k] - p.TargetDx[k];
                    float ey = o[2 * cells + k] - p.TargetDy[k];
                    loss += _opt.RegressionWeight * 0.5f * (ex * ex + ey * ey);
                    grad[cells + k] = _opt.RegressionWeight * ex;
                    grad[2 * cells + k] = _opt.RegressionWeight * ey;
                }
                else
                {
                    loss += Softplus(logit);
                    grad[k] = prob;
                }
            }

            for (int i = last; i >= 0; i--)
            {
                float[] input = i == 0 ? p.Input : outs[i - 1];
                (int inH, int inW) = i == 0 ? (S, S) : sizes[i - 1];
                float[]? gradIn = i == 0 ? null : new float[input.Length];
                LayerBackward(_net.Layers[i], input, inH, inW, outs[i], sizes[i].H, sizes[i].W,
                              grad, gradIn, g.W[i], g.B[i]);
                if (gradIn != null) grad = gradIn;
            }
            return loss;
        }

        private static float Softplus(float x) => x > 20 ? x : MathF.Log(1 + MathF.Exp(x));

        private static void LayerBackward(ConvLayer l, float[] input, int inH, int inW, float[] output,
                                          int outH, int outW, float[] gradOut, float[]? gradIn,
                                          float[] dW, float[] dB)
        {
            if (l.Relu)
                for (int i = 0; i < output.Length; i++)
                    if (output[i] <= 0f) gradOut[i] = 0f;

            int k = l.Kernel, s = l.Stride, p = l.Padding;
            int planeOut = outH * outW;
            for (int oc = 0; oc < l.OutChannels; oc++)
            {
                int outBase = oc * planeOut;
                float sb = 0;
                for (int i = 0; i < planeOut; i++) sb += gradOut[outBase + i];
                dB[oc] += sb;

                for (int ic = 0; ic < l.InChannels; ic++)
                {
                    int inBase = ic * inH * inW;
                    for (int ky = 0; ky < k; ky++)
                    {
                        int oyStart = Math.Max(0, ConvLayerMath.CeilDiv(p - ky, s));
                        int oyEnd = Math.Min(outH - 1, ConvLayerMath.FloorDiv(inH - 1 + p - ky, s));
                        for (int kx = 0; kx < k; kx++)
                        {
                            int oxStart = Math.Max(0, ConvLayerMath.CeilDiv(p - kx, s));
                            int oxEnd = Math.Min(outW - 1, ConvLayerMath.FloorDiv(inW - 1 + p - kx, s));
                            int wi = ((oc * l.InChannels + ic) * k + ky) * k + kx;
                            float w = l.Weights[wi];
                            float acc = 0;
                            for (int oy = oyStart; oy <= oyEnd; oy++)
                            {
                                int iy = oy * s - p + ky;
                                int inRow = inBase + iy * inW - p + kx;
                                int outRow = outBase + oy * outW;
                                for (int ox = oxStart; ox <= oxEnd; ox++)
                                {
                                    float go = gradOut[outRow + ox];
                                    if (go == 0f) continue;
                                    int ii = inRow + ox * s;
                                    acc += go * input[ii];
                                    if (gradIn != null) gradIn[ii] += w * go;
                                }
                            }
                            dW[wi] += acc;
                        }
                    }
                }
            }
        }

        private void ApplyAdam(Gradients g, float lr, float scale)
        {
            const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
            _t++;
            float c1 = 1f - MathF.Pow(beta1, _t), c2 = 1f - MathF.Pow(beta2, _t);
            for (int i = 0; i < _net.Layers.Count; i++)
            {
                ConvLayer l = _net.Layers[i];
                Update(l.Weights, g.W[i], _mW[i], _vW[i], _opt.WeightDecay);
                Update(l.Bias, g.B[i], _mB[i], _vB[i], 0f);
            }

            void Update(float[] param, float[] grad, float[] m, float[] v, float wd)
            {
                for (int j = 0; j < param.Length; j++)
                {
                    float gr = grad[j] * scale + wd * param[j];
                    m[j] = beta1 * m[j] + (1 - beta1) * gr;
                    v[j] = beta2 * v[j] + (1 - beta2) * gr * gr;
                    param[j] -= lr * (m[j] / c1) / (MathF.Sqrt(v[j] / c2) + eps);
                }
            }
        }

        private sealed class Gradients
        {
            public float[][] W { get; }
            public float[][] B { get; }
            public double Loss { get; set; }

            public Gradients(Network net)
            {
                W = net.Layers.Select(l => new float[l.Weights.Length]).ToArray();
                B = net.Layers.Select(l => new float[l.Bias.Length]).ToArray();
            }

            public void Add(Gradients o)
            {
                for (int i = 0; i < W.Length; i++)
                {
                    for (int j = 0; j < W[i].Length; j++) W[i][j] += o.W[i][j];
                    for (int j = 0; j < B[i].Length; j++) B[i][j] += o.B[i][j];
                }
            }
        }
    }

    internal static class ConvLayerMath
    {
        public static int CeilDiv(int a, int b) => a >= 0 ? (a + b - 1) / b : -((-a) / b);
        public static int FloorDiv(int a, int b) => a >= 0 ? a / b : -((-a + b - 1) / b);
    }
}
