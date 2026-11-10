using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace ImageFinder.Trainer
{
    public static class Program
    {
        public static int Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

            if (args.Length == 0 || args[0] is "-h" or "--help")
            {
                PrintUsage();
                return 1;
            }

            string dataDir = args[0];
            string output = "weights.bin";
            string? holdout = null;
            string? evaluate = null;
            var opt = new TrainOptions();
            int compositesPerTarget = 20;

            for (int i = 1; i < args.Length; i++)
            {
                string a = args[i];
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"Wert für {a} fehlt.");
                switch (a)
                {
                    case "--out": output = Next(); break;
                    case "--holdout": holdout = Next(); break;
                    case "--evaluate": evaluate = Next(); break;
                    case "--steps": opt.Steps = int.Parse(Next()); break;
                    case "--batch": opt.BatchSize = int.Parse(Next()); break;
                    case "--lr": opt.LearningRate = float.Parse(Next()); break;
                    case "--seed": opt.Seed = int.Parse(Next()); break;
                    case "--composites": compositesPerTarget = int.Parse(Next()); break;
                    default:
                        Console.WriteLine($"Unbekannte Option {a}");
                        PrintUsage();
                        return 1;
                }
            }

            Dataset data = Dataset.Load(dataDir, holdout);

            if (evaluate != null)
            {
                // Nur auswerten und Schwellwert neu kalibrieren, kein Training
                Network loaded = Network.Load(evaluate);
                List<EvalResult> ev = Evaluator.Evaluate(loaded, data.Train);
                loaded.Threshold = Evaluator.Calibrate(ev);
                Evaluator.Print("Auswertung", ev, loaded.Threshold);
                if (data.Holdout.Count > 0)
                    Evaluator.Print("Zurückgehaltene Bilder", Evaluator.Evaluate(loaded, data.Holdout), loaded.Threshold);
                loaded.Save(evaluate);
                Console.WriteLine($"Schwellwert in {evaluate} gespeichert.");
                return 0;
            }

            int realTrain = data.Train.Count;
            data.AddComposites(compositesPerTarget, new Random(opt.Seed + 1000));
            Console.WriteLine($"Trainingsbilder: {realTrain} echt " +
                              $"({data.Train.Count(s => !s.Synthetic && s.Objects.Count > 0)} mit Objekt), " +
                              $"{data.Train.Count - realTrain} synthetisch aus {data.Targets.Count} Targets " +
                              $"({string.Join(", ", data.TargetNames)})");
            if (data.Holdout.Count > 0)
                Console.WriteLine($"Zurückgehalten für den Test: {string.Join(", ", data.Holdout.Select(s => s.Name))}");

            Network net = Network.CreateDefault(opt.Seed);
            var trainer = new NetTrainer(net, opt);
            var sw = Stopwatch.StartNew();
            trainer.Train(data);
            Console.WriteLine($"Training fertig nach {sw.Elapsed:mm\\:ss}.");

            List<EvalResult> trainEval = Evaluator.Evaluate(net, data.Train.Where(s => !s.Synthetic));
            net.Threshold = Evaluator.Calibrate(trainEval);
            Evaluator.Print("Trainingsbilder", trainEval, net.Threshold);
            if (data.Holdout.Count > 0)
                Evaluator.Print("Zurückgehaltene Bilder (nie gesehen)", Evaluator.Evaluate(net, data.Holdout), net.Threshold);

            net.Save(output);
            Console.WriteLine();
            Console.WriteLine($"Gewichte gespeichert: {output}");
            return 0;
        }

        private static void PrintUsage()
        {
            Console.WriteLine("ImageFinder.Trainer <Datenordner> [Optionen]");
            Console.WriteLine();
            Console.WriteLine("Datenordner:  Test\\pos.txt  (\"name x y\" bzw. \"name - -\" für Bilder ohne Objekt)");
            Console.WriteLine("              Test\\name.png, Target\\*.png (Ausschnitte des Objekts, mittig)");
            Console.WriteLine();
            Console.WriteLine("  --out <datei>       Gewichtsdatei (Standard weights.bin)");
            Console.WriteLine("  --holdout <name>    Bild aus dem Training heraushalten und danach darauf testen");
            Console.WriteLine("  --evaluate <datei>  vorhandene Gewichte nur auswerten und Schwellwert neu setzen");
            Console.WriteLine("  --steps <n>         Trainingsschritte (Standard 3000)");
            Console.WriteLine("  --batch <n>         Ausschnitte pro Schritt (Standard 32)");
            Console.WriteLine("  --lr <x>            Lernrate (Standard 0.002)");
            Console.WriteLine("  --composites <n>    synthetische Bilder pro Target (Standard 20)");
            Console.WriteLine("  --seed <n>          Zufallsstartwert (Standard 1)");
        }
    }
}
