using System;
using System.Collections.Generic;

namespace ImageFinder.Trainer
{
    /// <summary>Ein Trainingsausschnitt samt Zielwerten pro Gitterzelle.</summary>
    public sealed class Patch
    {
        public Patch(int size, int cells)
        {
            Input = new float[3 * size * size];
            Label = new sbyte[cells * cells];
            TargetDx = new float[cells * cells];
            TargetDy = new float[cells * cells];
        }

        public float[] Input { get; }

        /// <summary>1 = Objekt, 0 = kein Objekt, -1 = ignorieren.</summary>
        public sbyte[] Label { get; }
        public float[] TargetDx { get; }
        public float[] TargetDy { get; }
    }

    /// <summary>
    /// Erzeugt zufällig augmentierte Ausschnitte (Größe, Drehung, Spiegelung, Farbe, Rauschen).
    /// </summary>
    public sealed class PatchSampler
    {
        public const int Size = 96;
        public const int CellSize = 8;
        public const int Cells = Size / CellSize;      // 12
        public const int Margin = 3;                    // Randzellen sehen Padding -> ignorieren

        private const float PosRadius = 6f;
        private const float IgnoreRadius = 13f;

        public float MinZoom { get; set; } = 0.6f;
        public float MaxZoom { get; set; } = 1.45f;

        public void Fill(Patch patch, Sample sample, Random rnd, (float X, float Y)? focus, bool positive)
        {
            float zoom = (float)Math.Exp(Lerp(Math.Log(MinZoom), Math.Log(MaxZoom), rnd.NextDouble()));
            float angle = (float)((rnd.NextDouble() * 2 - 1) * 10 * Math.PI / 180);
            bool flip = rnd.NextDouble() < 0.5;

            // Abbildung Patch -> Quelle: src = S0 + M * (p - P0)
            float cos = MathF.Cos(angle) / zoom, sin = MathF.Sin(angle) / zoom;
            float m00 = cos, m01 = -sin, m10 = sin, m11 = cos;
            if (flip) { m00 = -m00; m10 = -m10; }

            float p0x, p0y, s0x, s0y;
            RgbImage img = sample.Image;
            if (positive)
            {
                (float X, float Y) o = sample.Objects[rnd.Next(sample.Objects.Count)];
                float lo = Margin * CellSize - 4, hi = (Cells - Margin - 1) * CellSize + 4;
                p0x = Lerp(lo, hi, rnd.NextDouble());
                p0y = Lerp(lo, hi, rnd.NextDouble());
                s0x = o.X;
                s0y = o.Y;
            }
            else
            {
                p0x = p0y = Size / 2f;
                if (focus.HasValue)
                {
                    s0x = focus.Value.X + (float)(rnd.NextDouble() * 2 - 1) * 12;
                    s0y = focus.Value.Y + (float)(rnd.NextDouble() * 2 - 1) * 12;
                }
                else
                {
                    s0x = (float)(rnd.NextDouble() * img.Width);
                    s0y = (float)(rnd.NextDouble() * img.Height);
                }
            }

            // Farb-Augmentierung
            float contrast = Lerp(0.7f, 1.3f, rnd.NextDouble());
            float bright = Lerp(-0.2f, 0.2f, rnd.NextDouble());
            Span<float> gain = stackalloc float[3];
            for (int c = 0; c < 3; c++) gain[c] = Lerp(0.8f, 1.2f, rnd.NextDouble());
            float noise = (float)rnd.NextDouble() * 0.04f;
            bool gray = rnd.NextDouble() < 0.05;

            int plane = Size * Size;
            float[] input = patch.Input;
            for (int py = 0; py < Size; py++)
            {
                for (int px = 0; px < Size; px++)
                {
                    float dx = px - p0x, dy = py - p0y;
                    float sx = s0x + m00 * dx + m01 * dy;
                    float sy = s0y + m10 * dx + m11 * dy;
                    float r = Dataset.Bilinear(img, sx, sy, 0);
                    float g = Dataset.Bilinear(img, sx, sy, 1);
                    float b = Dataset.Bilinear(img, sx, sy, 2);
                    if (gray) r = g = b = 0.299f * r + 0.587f * g + 0.114f * b;
                    int i = py * Size + px;
                    input[i] = Augment(r, gain[0], contrast, bright, noise, rnd);
                    input[plane + i] = Augment(g, gain[1], contrast, bright, noise, rnd);
                    input[2 * plane + i] = Augment(b, gain[2], contrast, bright, noise, rnd);
                }
            }

            // Objektpositionen in Patch-Koordinaten: p = P0 + M^-1 (src - S0)
            float det = m00 * m11 - m01 * m10;
            float i00 = m11 / det, i01 = -m01 / det, i10 = -m10 / det, i11 = m00 / det;
            var objs = new List<(float X, float Y)>(sample.Objects.Count);
            foreach ((float X, float Y) o in sample.Objects)
            {
                float ex = o.X - s0x, ey = o.Y - s0y;
                objs.Add((p0x + i00 * ex + i01 * ey, p0y + i10 * ex + i11 * ey));
            }

            for (int r = 0; r < Cells; r++)
            {
                for (int c = 0; c < Cells; c++)
                {
                    int k = r * Cells + c;
                    patch.TargetDx[k] = patch.TargetDy[k] = 0;
                    if (r < Margin || c < Margin || r >= Cells - Margin || c >= Cells - Margin)
                    {
                        patch.Label[k] = -1;
                        continue;
                    }
                    float cx = c * CellSize, cy = r * CellSize;
                    float best = float.MaxValue;
                    (float X, float Y) nearest = default;
                    foreach ((float X, float Y) o in objs)
                    {
                        float d = MathF.Sqrt((o.X - cx) * (o.X - cx) + (o.Y - cy) * (o.Y - cy));
                        if (d < best) { best = d; nearest = o; }
                    }
                    if (best <= PosRadius)
                    {
                        patch.Label[k] = 1;
                        patch.TargetDx[k] = (nearest.X - cx) / CellSize;
                        patch.TargetDy[k] = (nearest.Y - cy) / CellSize;
                    }
                    else
                    {
                        patch.Label[k] = (sbyte)(best <= IgnoreRadius ? -1 : 0);
                    }
                }
            }
        }

        private static float Augment(float v, float gain, float contrast, float bright, float noise, Random rnd)
        {
            float x = v / 127.5f - 1f;
            x = x * gain;
            x = x * contrast + bright;
            if (noise > 0) x += (float)(rnd.NextDouble() * 2 - 1) * noise;
            return Math.Clamp(x, -1f, 1f);
        }

        private static float Lerp(float a, float b, double t) => (float)(a + (b - a) * t);
        private static double Lerp(double a, double b, double t) => a + (b - a) * t;
    }
}
