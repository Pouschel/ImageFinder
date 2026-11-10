using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Runtime.Versioning;

namespace ImageFinder
{
    /// <summary>Ein gefundenes Objekt: Mitte in Pixelkoordinaten des übergebenen Bildes und Konfidenz 0..1.</summary>
    public readonly record struct Detection(float X, float Y, float Score)
    {
        public Point ToPoint() => new Point((int)MathF.Round(X), (int)MathF.Round(Y));
    }

    /// <summary>
    /// Öffentliche Schnittstelle: findet das trainierte Objekt (z. B. den Angel-Schwimmer) in einem Bild.
    /// Thread-sicher: mehrere Threads dürfen dieselbe Instanz gleichzeitig benutzen.
    /// </summary>
    public sealed class ObjectFinder
    {
        private readonly Network _network;

        public ObjectFinder(Network network)
        {
            _network = network;
            Threshold = network.Threshold;
        }

        public static ObjectFinder Load(string weightsPath) => new ObjectFinder(Network.Load(weightsPath));
        public static ObjectFinder Load(Stream weights) => new ObjectFinder(Network.Load(weights));

        /// <summary>Mindest-Konfidenz für einen Treffer. Startwert kommt aus der Gewichtsdatei.</summary>
        public float Threshold { get; set; }

        /// <summary>Mindestabstand in Pixeln zwischen zwei gemeldeten Treffern.</summary>
        public float MinDistance { get; set; } = 20f;

        /// <summary>Rechnet auf mehreren Kernen. Für sehr kleine Ausschnitte kann false minimal schneller sein.</summary>
        public bool Parallel { get; set; } = true;

        /// <summary>Bester Treffer oder null, wenn nichts über dem Schwellwert liegt.</summary>
        [SupportedOSPlatform("windows")]
        public Detection? FindBest(Bitmap bitmap, Rectangle? region = null)
        {
            IReadOnlyList<Detection> all = FindAll(bitmap, region);
            return all.Count > 0 ? all[0] : null;
        }

        /// <summary>Alle Treffer über dem Schwellwert, nach Konfidenz absteigend sortiert.</summary>
        [SupportedOSPlatform("windows")]
        public IReadOnlyList<Detection> FindAll(Bitmap bitmap, Rectangle? region = null)
        {
            Rectangle rect = region ?? new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            rect.Intersect(new Rectangle(0, 0, bitmap.Width, bitmap.Height));
            RgbImage image = RgbImage.FromBitmap(bitmap, rect);
            return Offset(FindAll(image, Threshold), rect.X, rect.Y);
        }

        public Detection? FindBest(RgbImage image, Rectangle? region = null)
        {
            IReadOnlyList<Detection> all = FindAll(image, region);
            return all.Count > 0 ? all[0] : null;
        }

        public IReadOnlyList<Detection> FindAll(RgbImage image, Rectangle? region = null)
        {
            if (region == null)
                return FindAll(image, Threshold);
            Rectangle rect = region.Value;
            rect.Intersect(new Rectangle(0, 0, image.Width, image.Height));
            return Offset(FindAll(image.Crop(rect), Threshold), rect.X, rect.Y);
        }

        /// <summary>Alle lokalen Maxima über <paramref name="threshold"/> (auch für Auswertungen mit 0).</summary>
        public IReadOnlyList<Detection> FindAll(RgbImage image, float threshold)
        {
            ScoreMap map = _network.Evaluate(image, Parallel);
            var candidates = new List<Detection>();
            for (int r = 0; r < map.Rows; r++)
            {
                for (int c = 0; c < map.Cols; c++)
                {
                    float logit = map.Logit(r, c);
                    float score = ScoreMap.Sigmoid(logit);
                    if (score < threshold) continue;
                    if (!IsLocalMax(map, r, c, logit)) continue;
                    (float x, float y) = map.Center(r, c);
                    x = Math.Clamp(x, 0, image.Width - 1);
                    y = Math.Clamp(y, 0, image.Height - 1);
                    candidates.Add(new Detection(x, y, score));
                }
            }
            candidates.Sort((a, b) => b.Score.CompareTo(a.Score));

            // Non-Maximum-Suppression
            var result = new List<Detection>();
            float minD2 = MinDistance * MinDistance;
            foreach (Detection d in candidates)
            {
                bool tooClose = false;
                foreach (Detection kept in result)
                {
                    float dx = d.X - kept.X, dy = d.Y - kept.Y;
                    if (dx * dx + dy * dy < minD2) { tooClose = true; break; }
                }
                if (!tooClose) result.Add(d);
            }
            return result;
        }

        private static bool IsLocalMax(ScoreMap map, int r, int c, float v)
        {
            for (int dr = -1; dr <= 1; dr++)
            {
                for (int dc = -1; dc <= 1; dc++)
                {
                    if (dr == 0 && dc == 0) continue;
                    int rr = r + dr, cc = c + dc;
                    if (rr < 0 || cc < 0 || rr >= map.Rows || cc >= map.Cols) continue;
                    float o = map.Logit(rr, cc);
                    // bei Gleichstand gewinnt die erste Zelle
                    if (o > v || (o == v && (dr < 0 || (dr == 0 && dc < 0)))) return false;
                }
            }
            return true;
        }

        private static IReadOnlyList<Detection> Offset(IReadOnlyList<Detection> list, int ox, int oy)
        {
            if (ox == 0 && oy == 0) return list;
            var result = new List<Detection>(list.Count);
            foreach (Detection d in list) result.Add(d with { X = d.X + ox, Y = d.Y + oy });
            return result;
        }
    }
}
