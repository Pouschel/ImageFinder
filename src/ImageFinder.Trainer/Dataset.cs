using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace ImageFinder.Trainer
{
    /// <summary>Ein Trainings- oder Testbild mit den bekannten Objektmitten (leer = kein Objekt).</summary>
    public sealed class Sample
    {
        public Sample(string name, RgbImage image, List<(float X, float Y)> objects, bool synthetic)
        {
            Name = name;
            Image = image;
            Objects = objects;
            Synthetic = synthetic;
        }

        public string Name { get; }
        public RgbImage Image { get; }
        public List<(float X, float Y)> Objects { get; }
        public bool Synthetic { get; }
    }

    /// <summary>
    /// Lädt einen Datenordner:
    ///   Test/pos.txt   Zeilen "name x y" (Objektmitte) oder "name - -" (kein Objekt im Bild);
    ///                  mehrere Zeilen mit gleichem Namen = mehrere Objekte.
    ///   Test/name.png  die Bilder
    ///   Target/*.png   Ausschnitte des Objekts (mittig zugeschnitten), werden als
    ///                  zusätzliche synthetische Beispiele in andere Bilder eingeblendet.
    /// </summary>
    public sealed class Dataset
    {
        public List<Sample> Train { get; } = new List<Sample>();
        public List<Sample> Holdout { get; } = new List<Sample>();
        public List<RgbImage> Targets { get; } = new List<RgbImage>();
        public List<string> TargetNames { get; } = new List<string>();

        public static Dataset Load(string dataDir, string? holdout)
        {
            var ds = new Dataset();
            string testDir = Path.Combine(dataDir, "Test");
            string posFile = Path.Combine(testDir, "pos.txt");
            if (!File.Exists(posFile))
                throw new FileNotFoundException("pos.txt nicht gefunden.", posFile);

            var objects = new Dictionary<string, List<(float, float)>>(StringComparer.OrdinalIgnoreCase);
            var order = new List<string>();
            foreach (string rawLine in File.ReadAllLines(posFile))
            {
                string line = rawLine.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                string[] parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 3)
                    throw new FormatException($"pos.txt: Zeile '{line}' hat nicht drei Einträge.");
                string name = Path.GetFileNameWithoutExtension(parts[0]);
                if (!objects.TryGetValue(name, out List<(float, float)>? list))
                {
                    list = new List<(float, float)>();
                    objects[name] = list;
                    order.Add(name);
                }
                if (parts[1] != "-")
                    list.Add((float.Parse(parts[1], CultureInfo.InvariantCulture),
                              float.Parse(parts[2], CultureInfo.InvariantCulture)));
            }

            foreach (string name in order)
            {
                string file = Path.Combine(testDir, name + ".png");
                if (!File.Exists(file))
                {
                    Console.WriteLine($"Warnung: {file} fehlt, übersprungen.");
                    continue;
                }
                var sample = new Sample(name, RgbImage.LoadPng(file), objects[name], synthetic: false);
                bool isHoldout = holdout != null && string.Equals(name, holdout, StringComparison.OrdinalIgnoreCase);
                (isHoldout ? ds.Holdout : ds.Train).Add(sample);
            }

            foreach (string png in Directory.GetFiles(testDir, "*.png"))
            {
                string name = Path.GetFileNameWithoutExtension(png);
                if (!objects.ContainsKey(name))
                    Console.WriteLine($"Hinweis: {Path.GetFileName(png)} steht nicht in pos.txt und wird ignoriert.");
            }

            string targetDir = Path.Combine(dataDir, "Target");
            if (Directory.Exists(targetDir))
            {
                foreach (string png in Directory.GetFiles(targetDir, "*.png").OrderBy(f => f, StringComparer.Ordinal))
                {
                    string name = Path.GetFileNameWithoutExtension(png);
                    // Ein Target, das aus dem zurückgehaltenen Bild ausgeschnitten wurde ("3t" zu "3"), würde schummeln.
                    if (holdout != null && string.Equals(name, holdout + "t", StringComparison.OrdinalIgnoreCase))
                        continue;
                    ds.Targets.Add(RgbImage.LoadPng(png));
                    ds.TargetNames.Add(name);
                }
            }
            return ds;
        }

        /// <summary>
        /// Blendet die Target-Ausschnitte in zufällige Stellen der Trainingsbilder ein
        /// (weicher Rand, zufällige Größe). Ergebnis: kleine synthetische Bilder mit bekannter Mitte.
        /// </summary>
        public void AddComposites(int perTarget, Random rnd)
        {
            if (Train.Count == 0 || Targets.Count == 0) return;
            const int size = 200;
            List<Sample> bases = Train.Where(s => s.Image.Width >= size && s.Image.Height >= size).ToList();
            if (bases.Count == 0) return;

            for (int t = 0; t < Targets.Count; t++)
            {
                for (int k = 0; k < perTarget; k++)
                {
                    Sample b = bases[rnd.Next(bases.Count)];
                    int bx = 0, by = 0;
                    for (int tries = 0; tries < 50; tries++)
                    {
                        bx = rnd.Next(b.Image.Width - size + 1);
                        by = rnd.Next(b.Image.Height - size + 1);
                        // kein echtes Objekt im Ausschnitt
                        if (!b.Objects.Any(o => o.X > bx - 30 && o.X < bx + size + 30 && o.Y > by - 30 && o.Y < by + size + 30))
                            break;
                    }
                    RgbImage img = b.Image.Crop(new System.Drawing.Rectangle(bx, by, size, size));
                    float scale = (float)Math.Exp(rnd.NextDouble() * Math.Log(1.6) - Math.Log(1.6) / 2); // 0.79..1.26
                    float cx = 50 + (float)rnd.NextDouble() * (size - 100);
                    float cy = 50 + (float)rnd.NextDouble() * (size - 100);
                    Paste(img, Targets[t], cx, cy, scale);
                    Train.Add(new Sample($"{TargetNames[t]}#{k}", img,
                                         new List<(float, float)> { (cx, cy) }, synthetic: true));
                }
            }
        }

        private static void Paste(RgbImage dst, RgbImage src, float cx, float cy, float scale)
        {
            float w = src.Width * scale, h = src.Height * scale;
            int x0 = (int)Math.Floor(cx - w / 2), y0 = (int)Math.Floor(cy - h / 2);
            int x1 = (int)Math.Ceiling(cx + w / 2), y1 = (int)Math.Ceiling(cy + h / 2);
            const float feather = 3f;
            for (int y = Math.Max(0, y0); y < Math.Min(dst.Height, y1); y++)
            {
                for (int x = Math.Max(0, x0); x < Math.Min(dst.Width, x1); x++)
                {
                    float sx = (x + 0.5f - (cx - w / 2)) / scale - 0.5f;
                    float sy = (y + 0.5f - (cy - h / 2)) / scale - 0.5f;
                    if (sx < -0.5f || sy < -0.5f || sx > src.Width - 0.5f || sy > src.Height - 0.5f) continue;
                    // weicher Rand
                    float edge = Math.Min(Math.Min(sx + 0.5f, src.Width - 0.5f - sx), Math.Min(sy + 0.5f, src.Height - 0.5f - sy));
                    float alpha = Math.Clamp(edge * scale / feather, 0f, 1f);
                    int d = (y * dst.Width + x) * 3;
                    for (int ch = 0; ch < 3; ch++)
                    {
                        float v = Bilinear(src, sx, sy, ch);
                        dst.Pixels[d + ch] = (byte)Math.Clamp(alpha * v + (1 - alpha) * dst.Pixels[d + ch] + 0.5f, 0, 255);
                    }
                }
            }
        }

        public static float Bilinear(RgbImage img, float x, float y, int ch)
        {
            x = Math.Clamp(x, 0, img.Width - 1);
            y = Math.Clamp(y, 0, img.Height - 1);
            int ix = Math.Min((int)x, img.Width - 2 < 0 ? 0 : img.Width - 2);
            int iy = Math.Min((int)y, img.Height - 2 < 0 ? 0 : img.Height - 2);
            float fx = x - ix, fy = y - iy;
            int w = img.Width;
            byte[] p = img.Pixels;
            int ix1 = Math.Min(ix + 1, img.Width - 1), iy1 = Math.Min(iy + 1, img.Height - 1);
            float a = p[(iy * w + ix) * 3 + ch], b = p[(iy * w + ix1) * 3 + ch];
            float c = p[(iy1 * w + ix) * 3 + ch], d = p[(iy1 * w + ix1) * 3 + ch];
            return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
        }
    }
}
