using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace ImageFinder
{
    /// <summary>
    /// Minimaler PNG-Decoder für 8-Bit-Bilder (Graustufen, RGB, RGBA, Graustufen+Alpha, Palette),
    /// ohne Interlacing. Reicht für Screenshots und vermeidet die Abhängigkeit von System.Drawing.
    /// </summary>
    public static class PngReader
    {
        private static readonly byte[] Signature = { 137, 80, 78, 71, 13, 10, 26, 10 };

        public static RgbImage Read(Stream stream)
        {
            var sig = new byte[8];
            stream.ReadExactly(sig);
            if (!sig.AsSpan().SequenceEqual(Signature))
                throw new InvalidDataException("Keine PNG-Datei.");

            int width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
            byte[]? palette = null;
            using var idat = new MemoryStream();
            var header = new byte[8];

            while (true)
            {
                stream.ReadExactly(header);
                int length = BinaryPrimitives.ReadInt32BigEndian(header);
                string type = Encoding.ASCII.GetString(header, 4, 4);
                var data = new byte[length];
                stream.ReadExactly(data);
                stream.ReadExactly(new byte[4]); // CRC wird nicht geprüft

                if (type == "IHDR")
                {
                    width = BinaryPrimitives.ReadInt32BigEndian(data);
                    height = BinaryPrimitives.ReadInt32BigEndian(data.AsSpan(4));
                    bitDepth = data[8];
                    colorType = data[9];
                    interlace = data[12];
                }
                else if (type == "PLTE")
                {
                    palette = data;
                }
                else if (type == "IDAT")
                {
                    idat.Write(data);
                }
                else if (type == "IEND")
                {
                    break;
                }
            }

            if (bitDepth != 8)
                throw new NotSupportedException($"PNG-Bittiefe {bitDepth} wird nicht unterstützt (nur 8).");
            if (interlace != 0)
                throw new NotSupportedException("Interlaced PNG wird nicht unterstützt.");

            int channels = colorType switch
            {
                0 => 1, // Grau
                2 => 3, // RGB
                3 => 1, // Palette
                4 => 2, // Grau + Alpha
                6 => 4, // RGBA
                _ => throw new NotSupportedException($"PNG-Farbtyp {colorType} wird nicht unterstützt.")
            };
            if (colorType == 3 && palette == null)
                throw new InvalidDataException("PNG mit Palette, aber ohne PLTE-Chunk.");

            int stride = width * channels;
            var raw = new byte[(stride + 1) * height];
            idat.Position = 0;
            using (var z = new ZLibStream(idat, CompressionMode.Decompress))
            {
                z.ReadExactly(raw);
            }

            var cur = new byte[stride];
            var prev = new byte[stride];
            var image = new RgbImage(width, height);
            byte[] dst = image.Pixels;

            for (int y = 0; y < height; y++)
            {
                int filter = raw[y * (stride + 1)];
                Buffer.BlockCopy(raw, y * (stride + 1) + 1, cur, 0, stride);
                Unfilter(filter, cur, prev, channels);

                int d = y * width * 3;
                for (int x = 0; x < width; x++)
                {
                    switch (colorType)
                    {
                        case 0:
                        case 4:
                            byte g = cur[x * channels];
                            dst[d] = g; dst[d + 1] = g; dst[d + 2] = g;
                            break;
                        case 3:
                            int p = cur[x] * 3;
                            dst[d] = palette![p]; dst[d + 1] = palette[p + 1]; dst[d + 2] = palette[p + 2];
                            break;
                        default:
                            int s = x * channels;
                            dst[d] = cur[s]; dst[d + 1] = cur[s + 1]; dst[d + 2] = cur[s + 2];
                            break;
                    }
                    d += 3;
                }
                (cur, prev) = (prev, cur);
            }
            return image;
        }

        private static void Unfilter(int filter, byte[] cur, byte[] prev, int bpp)
        {
            int n = cur.Length;
            switch (filter)
            {
                case 0:
                    break;
                case 1: // Sub
                    for (int i = bpp; i < n; i++) cur[i] += cur[i - bpp];
                    break;
                case 2: // Up
                    for (int i = 0; i < n; i++) cur[i] += prev[i];
                    break;
                case 3: // Average
                    for (int i = 0; i < n; i++)
                    {
                        int left = i >= bpp ? cur[i - bpp] : 0;
                        cur[i] += (byte)((left + prev[i]) >> 1);
                    }
                    break;
                case 4: // Paeth
                    for (int i = 0; i < n; i++)
                    {
                        int a = i >= bpp ? cur[i - bpp] : 0;
                        int b = prev[i];
                        int c = i >= bpp ? prev[i - bpp] : 0;
                        int p = a + b - c;
                        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
                        int pred = pa <= pb && pa <= pc ? a : (pb <= pc ? b : c);
                        cur[i] += (byte)pred;
                    }
                    break;
                default:
                    throw new InvalidDataException($"Unbekannter PNG-Filter {filter}.");
            }
        }
    }
}
