using System;
using System.Numerics;
using System.Threading.Tasks;

namespace ImageFinder
{
    /// <summary>
    /// 2D-Faltungsschicht (quadratischer Kern, Padding = Kernel/2), optional mit ReLU.
    /// Daten liegen im Format Kanal × Zeile × Spalte (CHW) in einem float-Array.
    /// </summary>
    public sealed class ConvLayer
    {
        public int InChannels { get; }
        public int OutChannels { get; }
        public int Kernel { get; }
        public int Stride { get; }
        public int Padding => Kernel / 2;
        public bool Relu { get; }

        /// <summary>Gewichte, Index ((oc * InChannels + ic) * Kernel + ky) * Kernel + kx.</summary>
        public float[] Weights { get; }
        public float[] Bias { get; }

        public ConvLayer(int inChannels, int outChannels, int kernel, int stride, bool relu)
        {
            InChannels = inChannels;
            OutChannels = outChannels;
            Kernel = kernel;
            Stride = stride;
            Relu = relu;
            Weights = new float[outChannels * inChannels * kernel * kernel];
            Bias = new float[outChannels];
        }

        public int OutSize(int inSize) => (inSize + 2 * Padding - Kernel) / Stride + 1;

        /// <summary>
        /// Vorwärtsrechnung. <paramref name="output"/> muss mindestens OutChannels*outH*outW groß sein.
        /// </summary>
        public void Forward(float[] input, int inH, int inW, float[] output, bool parallel)
        {
            int outH = OutSize(inH), outW = OutSize(inW);
            if (parallel)
                Parallel.For(0, OutChannels, oc => ForwardChannel(oc, input, inH, inW, output, outH, outW));
            else
                for (int oc = 0; oc < OutChannels; oc++)
                    ForwardChannel(oc, input, inH, inW, output, outH, outW);
        }

        private void ForwardChannel(int oc, float[] input, int inH, int inW, float[] output, int outH, int outW)
        {
            int planeOut = outH * outW;
            int outBase = oc * planeOut;
            Array.Fill(output, Bias[oc], outBase, planeOut);
            int k = Kernel, s = Stride, p = Padding;
            Span<float> outSpan = output.AsSpan(outBase, planeOut);

            for (int ic = 0; ic < InChannels; ic++)
            {
                int inBase = ic * inH * inW;
                for (int ky = 0; ky < k; ky++)
                {
                    for (int kx = 0; kx < k; kx++)
                    {
                        float w = Weights[((oc * InChannels + ic) * k + ky) * k + kx];
                        // gültiger Bereich von ox, so dass ix = ox*s - p + kx in [0, inW)
                        int oxStart = Math.Max(0, CeilDiv(p - kx, s));
                        int oxEnd = Math.Min(outW - 1, FloorDiv(inW - 1 + p - kx, s));
                        if (oxEnd < oxStart) continue;
                        int oyStart = Math.Max(0, CeilDiv(p - ky, s));
                        int oyEnd = Math.Min(outH - 1, FloorDiv(inH - 1 + p - ky, s));
                        int count = oxEnd - oxStart + 1;

                        for (int oy = oyStart; oy <= oyEnd; oy++)
                        {
                            int iy = oy * s - p + ky;
                            int inRow = inBase + iy * inW + oxStart * s - p + kx;
                            int outRow = oy * outW + oxStart;
                            if (s == 1)
                                AxpyContiguous(w, input.AsSpan(inRow, count), outSpan.Slice(outRow, count));
                            else
                                for (int i = 0; i < count; i++)
                                    outSpan[outRow + i] += w * input[inRow + i * s];
                        }
                    }
                }
            }

            if (Relu)
                for (int i = 0; i < planeOut; i++)
                    if (outSpan[i] < 0f) outSpan[i] = 0f;
        }

        private static void AxpyContiguous(float w, ReadOnlySpan<float> x, Span<float> y)
        {
            int i = 0;
            if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
            {
                var vw = new Vector<float>(w);
                int n = x.Length - Vector<float>.Count;
                for (; i <= n; i += Vector<float>.Count)
                {
                    var vy = new Vector<float>(y.Slice(i)) + vw * new Vector<float>(x.Slice(i));
                    vy.CopyTo(y.Slice(i));
                }
            }
            for (; i < x.Length; i++) y[i] += w * x[i];
        }

        internal static int CeilDiv(int a, int b) => a >= 0 ? (a + b - 1) / b : -((-a) / b);
        internal static int FloorDiv(int a, int b) => a >= 0 ? a / b : -((-a + b - 1) / b);
    }
}
