using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.Versioning;

namespace ImageFinder
{
    /// <summary>
    /// Einfaches RGB-Bild (8 Bit pro Kanal, zeilenweise, R G B hintereinander).
    /// Plattformunabhängiges Austauschformat zwischen Bitmap, PNG-Dateien und dem Netz.
    /// </summary>
    public sealed class RgbImage
    {
        public int Width { get; }
        public int Height { get; }

        /// <summary>Pixel als R,G,B-Bytes, Länge = Width * Height * 3.</summary>
        public byte[] Pixels { get; }

        public RgbImage(int width, int height)
            : this(width, height, new byte[checked(width * height * 3)])
        {
        }

        public RgbImage(int width, int height, byte[] pixels)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Bildgröße muss positiv sein.");
            if (pixels.Length != width * height * 3)
                throw new ArgumentException("Pixel-Array hat die falsche Länge.", nameof(pixels));
            Width = width;
            Height = height;
            Pixels = pixels;
        }

        /// <summary>Lädt eine PNG-Datei ohne System.Drawing (läuft überall).</summary>
        public static RgbImage LoadPng(string path)
        {
            using FileStream fs = File.OpenRead(path);
            return PngReader.Read(fs);
        }

        public RgbImage Crop(Rectangle rect)
        {
            rect.Intersect(new Rectangle(0, 0, Width, Height));
            if (rect.Width <= 0 || rect.Height <= 0)
                throw new ArgumentException("Ausschnitt liegt außerhalb des Bildes.", nameof(rect));

            var result = new RgbImage(rect.Width, rect.Height);
            int rowBytes = rect.Width * 3;
            for (int y = 0; y < rect.Height; y++)
            {
                Buffer.BlockCopy(Pixels, ((rect.Y + y) * Width + rect.X) * 3,
                                 result.Pixels, y * rowBytes, rowBytes);
            }
            return result;
        }

        /// <summary>Kopiert einen Bereich einer System.Drawing.Bitmap in ein RgbImage.</summary>
        [SupportedOSPlatform("windows")]
        public static RgbImage FromBitmap(Bitmap bitmap, Rectangle? region = null)
        {
            Rectangle rect = region ?? new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            rect.Intersect(new Rectangle(0, 0, bitmap.Width, bitmap.Height));
            if (rect.Width <= 0 || rect.Height <= 0)
                throw new ArgumentException("Ausschnitt liegt außerhalb der Bitmap.", nameof(region));

            var result = new RgbImage(rect.Width, rect.Height);
            BitmapData data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte[] dst = result.Pixels;
                    for (int y = 0; y < rect.Height; y++)
                    {
                        byte* src = (byte*)data.Scan0 + (long)y * data.Stride;
                        int d = y * rect.Width * 3;
                        for (int x = 0; x < rect.Width; x++)
                        {
                            // Format24bppRgb liegt im Speicher als B,G,R vor.
                            dst[d++] = src[2];
                            dst[d++] = src[1];
                            dst[d++] = src[0];
                            src += 3;
                        }
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(data);
            }
            return result;
        }
    }
}
