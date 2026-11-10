using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.Runtime.Versioning;

namespace ImageFinder.Cli
{
    /// <summary>
    /// Sucht das trainierte Objekt in einem Bild und misst die Laufzeit.
    /// Unter Windows über System.Drawing.Bitmap (wie in eigenen Programmen), sonst über den PNG-Reader.
    /// </summary>
    public static class Program
    {
        public static int Main(string[] args)
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            if (args.Length < 2)
            {
                Console.WriteLine("ImageFinder.Cli <weights.bin> <bild.png> [x y breite höhe] [--repeat n]");
                return 1;
            }

            ObjectFinder finder = ObjectFinder.Load(args[0]);
            string imagePath = args[1];
            Rectangle? region = null;
            int repeat = 10;
            var rest = new List<string>(args[2..]);
            int ri = rest.IndexOf("--repeat");
            if (ri >= 0)
            {
                repeat = int.Parse(rest[ri + 1]);
                rest.RemoveRange(ri, 2);
            }
            if (rest.Count == 4)
                region = new Rectangle(int.Parse(rest[0]), int.Parse(rest[1]), int.Parse(rest[2]), int.Parse(rest[3]));

            Console.WriteLine($"Schwellwert: {finder.Threshold:F3}");

            Func<IReadOnlyList<Detection>> run;
            if (OperatingSystem.IsWindowsVersionAtLeast(6, 1))
            {
                run = BitmapRunner(finder, imagePath, region);
            }
            else
            {
                RgbImage image = RgbImage.LoadPng(imagePath);
                run = () => finder.FindAll(image, region);
            }

            IReadOnlyList<Detection> result = run(); // Aufwärmen (JIT)
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < repeat; i++) result = run();
            sw.Stop();

            if (result.Count == 0)
                Console.WriteLine("Kein Objekt gefunden.");
            foreach (Detection d in result)
                Console.WriteLine($"Treffer bei ({d.X:F1}, {d.Y:F1})  Konfidenz {d.Score:F3}");
            Console.WriteLine($"Laufzeit: {sw.Elapsed.TotalMilliseconds / repeat:F1} ms pro Bild");
            return 0;
        }

        [SupportedOSPlatform("windows6.1")]
        private static Func<IReadOnlyList<Detection>> BitmapRunner(ObjectFinder finder, string path, Rectangle? region)
        {
            var bitmap = new Bitmap(path);
            return () => finder.FindAll(bitmap, region);
        }
    }
}
