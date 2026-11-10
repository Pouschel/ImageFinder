using System;
using System.Collections.Generic;
using System.Linq;

namespace ImageFinder.Trainer
{
    public sealed class EvalResult
    {
        public string Name { get; set; } = "";
        public bool HasObject { get; set; }
        public float PositiveScore { get; set; }   // beste Konfidenz nahe der echten Mitte
        public float PositionError { get; set; }   // Abstand dieser Detektion zur echten Mitte
        public float MaxFalseScore { get; set; }   // beste Konfidenz weit weg von jedem Objekt
        public (float X, float Y) Top { get; set; }
        public float TopScore { get; set; }
    }

    public static class Evaluator
    {
        public const float HitRadius = 12f;

        public static List<EvalResult> Evaluate(Network net, IEnumerable<Sample> samples)
        {
            var finder = new ObjectFinder(net) { MinDistance = 16 };
            var results = new List<EvalResult>();
            foreach (Sample s in samples)
            {
                IReadOnlyList<Detection> dets = finder.FindAll(s.Image, 0f);
                var r = new EvalResult { Name = s.Name, HasObject = s.Objects.Count > 0 };
                if (dets.Count > 0)
                {
                    r.Top = (dets[0].X, dets[0].Y);
                    r.TopScore = dets[0].Score;
                }
                foreach (Detection d in dets)
                {
                    float near = s.Objects.Count == 0 ? float.MaxValue
                        : s.Objects.Min(o => MathF.Sqrt((o.X - d.X) * (o.X - d.X) + (o.Y - d.Y) * (o.Y - d.Y)));
                    if (near <= HitRadius)
                    {
                        if (d.Score > r.PositiveScore) { r.PositiveScore = d.Score; r.PositionError = near; }
                    }
                    else if (near > 20f)
                    {
                        r.MaxFalseScore = Math.Max(r.MaxFalseScore, d.Score);
                    }
                }
                results.Add(r);
            }
            return results;
        }

        /// <summary>Bester Treffer des Bildes ist das echte Objekt?</summary>
        public static bool TopIsObject(EvalResult r) => r.HasObject && r.PositiveScore > 0 && r.PositiveScore >= r.TopScore;

        public static void Print(string title, List<EvalResult> results, float threshold)
        {
            Console.WriteLine();
            Console.WriteLine($"== {title} (Schwellwert {threshold:F3}) ==");
            Console.WriteLine($"{"Bild",-10} {"Objekt",6} {"Score Ziel",10} {"Fehler px",9} {"max. Fehlalarm",14}  Ergebnis FindBest");
            foreach (EvalResult r in results)
            {
                string verdict;
                if (r.HasObject)
                    verdict = TopIsObject(r) && r.PositiveScore >= threshold ? "OK"
                            : TopIsObject(r) ? "unter Schwellwert"
                            : "FALSCH (Fehlalarm vorn)";
                else
                    verdict = r.TopScore >= threshold ? "FEHLALARM" : "OK";
                if (r.MaxFalseScore >= threshold && verdict == "OK" && r.HasObject)
                    verdict += " (weitere Fehlalarme in FindAll)";
                string pos = r.HasObject ? $"{r.PositiveScore,10:F3}" : $"{"-",10}";
                string err = r.HasObject && r.PositiveScore > 0 ? $"{r.PositionError,9:F1}" : $"{"-",9}";
                Console.WriteLine($"{r.Name,-10} {(r.HasObject ? "ja" : "nein"),6} {pos} {err} {r.MaxFalseScore,14:F3}  {verdict}");
            }
        }

        /// <summary>
        /// Schwellwert für FindBest: zwischen schwächstem Objekt-Treffer (wenn er Platz 1 ist)
        /// und dem stärksten Treffer in Bildern ohne Objekt bzw. Fehlalarmen, die vor dem Objekt liegen.
        /// Mittelpunkt im Logit-Raum.
        /// </summary>
        public static float Calibrate(List<EvalResult> results)
        {
            float minPos = results.Where(TopIsObject).Select(r => r.PositiveScore).DefaultIfEmpty(1f).Min();
            float maxNeg = results.Where(r => !r.HasObject).Select(r => r.TopScore)
                .Concat(results.Where(r => r.HasObject && !TopIsObject(r)).Select(r => r.TopScore))
                .DefaultIfEmpty(0f).Max();
            if (minPos <= maxNeg)
            {
                Console.WriteLine($"Warnung: Objekt-Treffer ({minPos:F3}) und Fehlalarme ({maxNeg:F3}) überlappen, nehme 0.5.");
                return 0.5f;
            }
            float l = (Logit(minPos) + Logit(maxNeg)) / 2;
            return ScoreMap.Sigmoid(l);
        }

        private static float Logit(float p)
        {
            p = Math.Clamp(p, 1e-6f, 1 - 1e-6f);
            return MathF.Log(p / (1 - p));
        }
    }
}
