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

### Parameter

```
ImageFinder.Trainer <Datenordner> [Optionen]
```

| Parameter | Standard | Kurz |
|---|---|---|
| `<Datenordner>` | – | Ordner mit `Test\` und `Target\`, Pflichtangabe |
| `--out <datei>` | `weights.bin` | Ausgabedatei für die Gewichte |
| `--steps <n>` | 3000 | Anzahl der Trainingsschritte |
| `--batch <n>` | 32 | Ausschnitte pro Schritt |
| `--lr <x>` | 0.002 (mit `--init` 0.0005) | Lernrate |
| `--init <datei>` | – | mit vorhandenen Gewichten weitertrainieren |
| `--holdout <name>` | – | ein Bild nicht trainieren, nur darauf testen |
| `--composites <n>` | 20 | synthetische Bilder pro Target |
| `--seed <n>` | 1 | Zufallsstartwert |
| `--evaluate <datei>` | – | nicht trainieren, nur auswerten und Schwellwert neu setzen |

**`<Datenordner>`**
Erster Parameter, ohne Bindestrich. Erwartet `Test\pos.txt`, die zugehörigen PNG-Bilder in `Test\`
und optional `Target\*.png` (Aufbau siehe oben). Bilder in `Test\`, die nicht in `pos.txt` stehen,
werden mit einem Hinweis übersprungen; Einträge in `pos.txt` ohne Bild erzeugen eine Warnung.

**`--out <datei>`**
Wohin die fertigen Gewichte geschrieben werden, relativ zum aktuellen Verzeichnis oder absolut.
Eine vorhandene Datei wird ohne Nachfrage überschrieben. Die Datei enthält Netzaufbau, Gewichte und
den kalibrierten Schwellwert, also alles, was `ObjectFinder.Load` braucht.

**`--steps <n>`**
Ein Schritt = ein Batch Ausschnitte vorwärts und rückwärts rechnen und die Gewichte einmal anpassen.
Die Dauer pro Schritt hängt nur von `--batch` und der Netzgröße ab, nicht von der Anzahl der Bilder
(auf 2 Kernen etwa 0,3 s). Alle 50 Schritte wird der Loss ausgegeben, alle 250 Schritte läuft die
Fehlalarm-Suche über alle echten Bilder.
Richtwerte: 3000 für ein Training von vorn, 500–1500 beim Weitertrainieren mit `--init`.
Mehr Schritte helfen, solange der Loss noch sinkt; danach bringt mehr Bildmaterial mehr als mehr Schritte.

**`--batch <n>`**
Wie viele zufällige Ausschnitte (96×96 Pixel) pro Schritt verarbeitet werden. Etwa 40 % enthalten
das Objekt, der Rest ist Hintergrund, davon rund die Hälfte gezielt aus gesammelten Fehlalarmen.
Größere Batches machen das Training ruhiger, aber jeder Schritt dauert entsprechend länger.
Die Standard-Lernrate ist auf 32 abgestimmt; bei deutlich anderen Werten ggf. `--lr` anpassen.

**`--lr <x>`**
Lernrate des Adam-Optimierers: wie stark die Gewichte pro Schritt verändert werden.
Die ersten 100 Schritte wird sie langsam hochgefahren, danach fällt sie bis zum Ende auf 5 % ab
(Kosinus-Verlauf); die Ausgabe zeigt den aktuellen Wert.
Zu groß: Loss springt oder wird `NaN`. Zu klein: Loss sinkt kaum.
Ohne Angabe 0.002 bzw. 0.0005 mit `--init`; ein eigener Wert gilt in beiden Fällen.

**`--init <datei>`**
Startet mit den Gewichten aus der Datei statt mit Zufallswerten. Der Netzaufbau kommt aus der Datei.
Gleichzeitig wird die Lernrate auf 0.0005 gesenkt (sofern kein `--lr` angegeben ist) und die
Fehlalarm-Suche läuft schon vor dem ersten Schritt. Der interne Zustand des Optimierers wird nicht
gespeichert und beginnt neu, daher wieder die 100 Schritte Anlaufphase.
Immer mit allen Bildern weitertrainieren, nicht nur mit den neuen, sonst vergisst das Netz das alte Material.
`--out` darf dieselbe Datei sein; eine neue Datei erlaubt den Vergleich mit `--evaluate`.

**`--holdout <name>`**
Name eines Bildes aus `pos.txt` (ohne `.png`). Dieses Bild wird nicht trainiert, sondern am Ende
getrennt ausgewertet („Zurückgehaltene Bilder“). Zusätzlich wird ein Target `<name>t.png` weggelassen,
weil es aus diesem Bild geschnitten wäre. So sieht man, wie das Netz auf ein nie gesehenes Bild reagiert;
die Tabelle der Trainingsbilder ist dafür zu optimistisch.
Der Schwellwert wird nur aus den Trainingsbildern bestimmt. Die gespeicherten Gewichte kennen das
zurückgehaltene Bild nicht – für den Einsatz danach ohne `--holdout` trainieren.

**`--composites <n>`**
Für jedes Target werden n kleine Bilder (200×200) erzeugt: ein zufälliger Ausschnitt eines
Trainingsbildes ohne echtes Objekt, in den das Target mit weichem Rand und zufälliger Größe
(0,79- bis 1,26-fach) eingeblendet wird. Das zeigt das Objekt vor zusätzlichen Hintergründen.
Echte Testbilder sind wertvoller; `0` schaltet die Targets ganz ab.

**`--seed <n>`**
Startwert für alle Zufallszahlen (Anfangsgewichte, Ausschnitte, Augmentierung, Composites).
Gleicher Seed und gleiche Daten ergeben dasselbe Ergebnis. Verschiedene Seeds zeigen, wie stark das
Ergebnis vom Zufall abhängt; bei wenigen Bildern kann das merklich sein.

**`--evaluate <datei>`**
Kein Training. Lädt die Gewichte, wertet alle Bilder aus `pos.txt` aus, bestimmt den Schwellwert neu
und schreibt ihn **in dieselbe Datei** zurück (die Gewichte selbst bleiben unverändert).
Nützlich nach dem Hinzufügen neuer Bilder oder um zwei Gewichtsdateien zu vergleichen.
Mit `--holdout` wird das genannte Bild getrennt aufgeführt und nicht zur Kalibrierung benutzt.
Alle anderen Trainingsoptionen werden dabei ignoriert.

### Beispiele

```
rem Training von vorn
ImageFinder.Trainer G:\Sync\Bilder\ImageFinder\Angler --out weights\angler.bin

rem Test: wie gut wird Bild 9 erkannt, wenn das Netz es nie gesehen hat?
ImageFinder.Trainer G:\Sync\Bilder\ImageFinder\Angler --holdout 9 --out test_ohne9.bin

rem neue Bilder dazu, weitertrainieren
ImageFinder.Trainer G:\Sync\Bilder\ImageFinder\Angler --init weights\angler.bin --steps 1000 --out weights\angler_neu.bin

rem vorhandene Gewichte auf dem aktuellen Bildbestand prüfen
ImageFinder.Trainer G:\Sync\Bilder\ImageFinder\Angler --evaluate weights\angler_neu.bin
```

### Feste Einstellungen im Code

Diese Werte sind keine Parameter, lassen sich aber in `TrainOptions` (`NetTrainer.cs`) bzw.
`PatchSampler.cs` ändern:

| Wert | Standard | Bedeutung |
|---|---|---|
| `PositiveFraction` | 0.4 | Anteil der Ausschnitte mit Objekt |
| `HardNegativeFraction` | 0.5 | Anteil gesammelter Fehlalarme unter den Hintergrund-Ausschnitten |
| `HardMiningInterval` | 250 | alle wie viele Schritte nach Fehlalarmen gesucht wird |
| Pool-Größe | 3000 | maximale Zahl gemerkter Fehlalarme (älteste fliegen raus), `MineHardNegatives` |
| `PositiveWeight` | 3 | Gewicht der Objekt-Zellen im Loss |
| `RegressionWeight` | 2 | Gewicht des Mittelpunkt-Versatzes im Loss |
| `WeightDecay` | 0.0001 | leichte Bremse gegen zu große Gewichte |
| `MinZoom` / `MaxZoom` | 0.6 / 1.45 | Größenbereich der Augmentierung |

Weitere Augmentierung in `PatchSampler.Fill`: Drehung ±10°, Spiegelung (50 %), Kontrast 0,7–1,3,
Helligkeit ±0,2, Farbkanäle je 0,8–1,2, Rauschen bis 0,04, 5 % Graustufen.

### Weitertrainieren mit neuen Bildern

```
dotnet run -c Release --project src\ImageFinder.Trainer -- G:\Sync\Bilder\ImageFinder\Angler --init weights\angler.bin --steps 1000 --out weights\angler.bin
```

Neue Bilder einfach in `Test` legen und in `pos.txt` eintragen, dann mit `--init` über **alle** Bilder
weitertrainieren. Nur mit den neuen Bildern zu trainieren würde das Netz das alte Material vergessen lassen.
Mit `--init` sinkt die Lernrate auf 0.0005 (änderbar mit `--lr`), und die Fehlalarm-Suche läuft sofort zu Beginn.
Nach größeren Änderungen am Bildbestand ab und zu von vorn trainieren.

Die Dauer eines Schritts hängt nicht von der Bildanzahl ab (pro Schritt werden 32 Ausschnitte gezogen).
Nur die Fehlalarm-Suche alle 250 Schritte läuft über alle Bilder und wird mit mehr Bildern länger.

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
