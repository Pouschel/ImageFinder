using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace ImageFinder
{
    /// <summary>
    /// Kleines voll-konvolutionales Netz. Ausgabe pro Gitterzelle (Abstand = <see cref="CellSize"/> Pixel):
    /// Kanal 0 = Logit "Objekt hier", Kanal 1/2 = Versatz der Objektmitte in Zellen (dx, dy).
    /// Die Zelle (r, c) liegt bei Pixel (c * CellSize, r * CellSize).
    /// </summary>
    public sealed class Network
    {
        private const string Magic = "IFNN";
        private const int FormatVersion = 1;

        public List<ConvLayer> Layers { get; } = new List<ConvLayer>();

        /// <summary>Empfohlener Schwellwert für die Objektwahrscheinlichkeit (vom Trainer kalibriert).</summary>
        public float Threshold { get; set; } = 0.5f;

        public int CellSize
        {
            get
            {
                int s = 1;
                foreach (ConvLayer l in Layers) s *= l.Stride;
                return s;
            }
        }

        /// <summary>Standard-Architektur: 3 Stride-2-Schichten (Zellgröße 8 px), Rezeptives Feld 47 px.</summary>
        public static Network CreateDefault(int seed)
        {
            var net = new Network();
            net.Layers.Add(new ConvLayer(3, 16, 3, 2, true));
            net.Layers.Add(new ConvLayer(16, 24, 3, 2, true));
            net.Layers.Add(new ConvLayer(24, 32, 3, 2, true));
            net.Layers.Add(new ConvLayer(32, 32, 3, 1, true));
            net.Layers.Add(new ConvLayer(32, 32, 3, 1, true));
            net.Layers.Add(new ConvLayer(32, 3, 1, 1, false));

            var rnd = new Random(seed);
            foreach (ConvLayer l in net.Layers)
            {
                // He-Initialisierung
                double std = Math.Sqrt(2.0 / (l.InChannels * l.Kernel * l.Kernel));
                for (int i = 0; i < l.Weights.Length; i++)
                    l.Weights[i] = (float)(Gaussian(rnd) * std);
            }
            ConvLayer head = net.Layers[^1];
            for (int i = 0; i < head.Weights.Length; i++) head.Weights[i] *= 0.1f;
            head.Bias[0] = -4f; // Startwahrscheinlichkeit "Objekt" klein
            return net;
        }

        private static double Gaussian(Random rnd)
        {
            double u1 = 1.0 - rnd.NextDouble(), u2 = rnd.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }

        /// <summary>Wandelt ein RGB-Bild in den Netz-Eingang (CHW, Werte in [-1, 1]).</summary>
        public static float[] ToInput(RgbImage image)
        {
            int n = image.Width * image.Height;
            var input = new float[3 * n];
            byte[] px = image.Pixels;
            const float scale = 1f / 127.5f;
            for (int i = 0; i < n; i++)
            {
                input[i] = px[3 * i] * scale - 1f;
                input[n + i] = px[3 * i + 1] * scale - 1f;
                input[2 * n + i] = px[3 * i + 2] * scale - 1f;
            }
            return input;
        }

        /// <summary>
        /// Vorwärtsrechnung, liefert die Ausgaben aller Schichten (für das Training).
        /// sizes[i] = (Höhe, Breite) der Ausgabe von Schicht i.
        /// </summary>
        public float[][] ForwardAll(float[] input, int height, int width, bool parallel, out (int H, int W)[] sizes)
        {
            var outputs = new float[Layers.Count][];
            sizes = new (int, int)[Layers.Count];
            float[] cur = input;
            int h = height, w = width;
            for (int i = 0; i < Layers.Count; i++)
            {
                ConvLayer l = Layers[i];
                int oh = l.OutSize(h), ow = l.OutSize(w);
                var o = new float[l.OutChannels * oh * ow];
                l.Forward(cur, h, w, o, parallel);
                outputs[i] = o;
                sizes[i] = (oh, ow);
                cur = o;
                h = oh;
                w = ow;
            }
            return outputs;
        }

        public ScoreMap Evaluate(RgbImage image, bool parallel = true)
        {
            float[] input = ToInput(image);
            float[] cur = input;
            int h = image.Height, w = image.Width;
            foreach (ConvLayer l in Layers)
            {
                int oh = l.OutSize(h), ow = l.OutSize(w);
                var o = new float[l.OutChannels * oh * ow];
                l.Forward(cur, h, w, o, parallel);
                cur = o;
                h = oh;
                w = ow;
            }
            return new ScoreMap(cur, h, w, CellSize);
        }

        public void Save(string path)
        {
            using FileStream fs = File.Create(path);
            Save(fs);
        }

        public void Save(Stream stream)
        {
            using var bw = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
            bw.Write(Encoding.ASCII.GetBytes(Magic));
            bw.Write(FormatVersion);
            bw.Write(Threshold);
            bw.Write(Layers.Count);
            foreach (ConvLayer l in Layers)
            {
                bw.Write(l.InChannels);
                bw.Write(l.OutChannels);
                bw.Write(l.Kernel);
                bw.Write(l.Stride);
                bw.Write(l.Relu);
                foreach (float f in l.Weights) bw.Write(f);
                foreach (float f in l.Bias) bw.Write(f);
            }
        }

        public static Network Load(string path)
        {
            using FileStream fs = File.OpenRead(path);
            return Load(fs);
        }

        public static Network Load(Stream stream)
        {
            using var br = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            if (Encoding.ASCII.GetString(br.ReadBytes(4)) != Magic)
                throw new InvalidDataException("Keine ImageFinder-Gewichtsdatei.");
            int version = br.ReadInt32();
            if (version != FormatVersion)
                throw new InvalidDataException($"Gewichtsdatei-Version {version} wird nicht unterstützt.");

            var net = new Network { Threshold = br.ReadSingle() };
            int count = br.ReadInt32();
            for (int i = 0; i < count; i++)
            {
                int inC = br.ReadInt32(), outC = br.ReadInt32(), k = br.ReadInt32(), s = br.ReadInt32();
                bool relu = br.ReadBoolean();
                var l = new ConvLayer(inC, outC, k, s, relu);
                for (int j = 0; j < l.Weights.Length; j++) l.Weights[j] = br.ReadSingle();
                for (int j = 0; j < l.Bias.Length; j++) l.Bias[j] = br.ReadSingle();
                net.Layers.Add(l);
            }
            return net;
        }
    }

    /// <summary>Rohausgabe des Netzes über dem Zellgitter.</summary>
    public sealed class ScoreMap
    {
        private readonly float[] _data;

        public int Rows { get; }
        public int Cols { get; }
        public int CellSize { get; }

        internal ScoreMap(float[] data, int rows, int cols, int cellSize)
        {
            _data = data;
            Rows = rows;
            Cols = cols;
            CellSize = cellSize;
        }

        public float Logit(int r, int c) => _data[r * Cols + c];
        public float Score(int r, int c) => Sigmoid(Logit(r, c));

        /// <summary>Geschätzte Objektmitte (in Pixeln des ausgewerteten Bildes) für Zelle (r, c).</summary>
        public (float X, float Y) Center(int r, int c)
        {
            int plane = Rows * Cols;
            float dx = Math.Clamp(_data[plane + r * Cols + c], -1.5f, 1.5f);
            float dy = Math.Clamp(_data[2 * plane + r * Cols + c], -1.5f, 1.5f);
            return ((c + dx) * CellSize, (r + dy) * CellSize);
        }

        public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    }
}
