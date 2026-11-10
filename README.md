# ImageFinder

Findet ein trainiertes Objekt (z. B. den Angel-Schwimmer) in Screenshots.
Kleines voll-konvolutionales neuronales Netz in reinem C#, keine ML-Bibliotheken.

## Projekte

| Projekt | Zweck |
|---|---|
| `ImageFinder.Core` | Bibliothek zum Einbinden: Netz, Gewichtsdatei, `ObjectFinder`, PNG-Reader |
| `ImageFinder.Trainer` | Trainiert das Netz aus einem Datenordner und schreibt `weights.bin` |
| `ImageFinder.Cli` | Sucht das Objekt in einem Bild und misst die Laufzeit |

Einzige Paketabhängigkeit: `System.Drawing.Common` (für die `Bitmap`-Schnittstelle, Windows).

## Datenordner

```
Angler\
  Test\pos.txt      eine Zeile pro Objekt:  name x y   (Objektmitte in Pixeln)
                    Bild ohne Objekt:       name - -
  Test\name.png     Screenshots
  Target\*.png      eng um das Objekt zugeschnittene Ausschnitte (Objekt mittig)
```

Die Target-Ausschnitte werden beim Training zusätzlich in zufällige Stellen anderer Bilder
eingeblendet. Ein Target `3t.png` gilt als aus `3.png` geschnitten und wird bei `--holdout 3` weggelassen.

## Training

```
dotnet run -c Release --project src\ImageFinder.Trainer -- G:\Sync\Bilder\ImageFinder\Angler --out weights.bin
```

Optionen: `--steps` (3000), `--batch` (32), `--holdout <name>` (Bild nicht trainieren, nur testen),
`--composites` (20 synthetische Bilder pro Target), `--seed`, `--lr`,
`--evaluate <weights.bin>` (nicht trainieren, nur vorhandene Gewichte auswerten und Schwellwert neu setzen).

Am Ende steht eine Tabelle: Konfidenz am echten Objekt, Positionsfehler, stärkster Fehlalarm
und ob `FindBest` das richtige Ergebnis liefert. Der Schwellwert wird so gelegt, dass der beste
Treffer in Bildern mit Objekt darüber und der beste Treffer in Bildern ohne Objekt darunter liegt,
und in `weights.bin` gespeichert.

Jeder Screenshot mit Schwimmer ist gleichzeitig Negativmaterial: alles außerhalb der Mitte
wird als „kein Objekt“ trainiert, Fehlalarme werden automatisch gesammelt und gezielt nachtrainiert.

Aussagekräftig ist vor allem `--holdout`: nur so sieht man, ob ein nie gesehenes Bild erkannt wird.

## Benutzung der Bibliothek

```csharp
using System.Drawing;
using ImageFinder;

ObjectFinder finder = ObjectFinder.Load("weights.bin");

using var bmp = new Bitmap("screenshot.png");
Detection? hit = finder.FindBest(bmp);                                  // ganzes Bild
Detection? hit2 = finder.FindBest(bmp, new Rectangle(400, 300, 100, 100)); // nur Ausschnitt

if (hit is Detection d)
    Console.WriteLine($"Schwimmer bei {d.X:F0},{d.Y:F0} (Konfidenz {d.Score:F2})");
```

Koordinaten beziehen sich immer auf die ganze Bitmap und zeigen auf die Objektmitte.
`FindAll` liefert alle Treffer, `Threshold` und `MinDistance` sind einstellbar.
`ObjectFinder` ist thread-sicher.

Laufzeit (gemessen auf 2 Kernen): Vollbild ~1100×700 etwa 130–290 ms, Ausschnitt 100×100 etwa 5 ms.

## Netz

Eingang RGB, 6 Faltungsschichten (3×3, Stride 2/2/2/1/1, dann 1×1), etwa 29 000 Gewichte (16/24/32/32/32 Kanäle).
Ausgabe auf einem 8-Pixel-Gitter: Wahrscheinlichkeit „Objekt hier“ und Versatz der Mitte innerhalb der Zelle.
Rezeptives Feld 47 Pixel; trainiert für Objektgrößen von etwa 0,6- bis 1,45-fach der Beispiele.
