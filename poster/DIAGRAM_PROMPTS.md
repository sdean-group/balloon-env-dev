# Diagram production briefs for the wind poster

These briefs correspond exactly to the numbered artwork slots in `poster.tex`. Export
final artwork as high-resolution PNG or PDF with no outer whitespace, then replace the
matching `\artslot` command with `\includegraphics`. Keep labels editable when possible.

## Shared art direction

- Format: scientific conference poster graphic, clean flat vector style, white or very
  pale gray background, no decorative texture, no photorealism, no drop shadows.
- Palette: navy `#173B6C`, teal `#198F83`, orange `#E28B24`, muted gray `#5F6E82`, pale
  teal `#E5F4F1`, pale orange `#FFF3E2`; use red `#D74A4A` only for zoom boxes or seams.
- Typography: Lato or a neutral sans serif. Labels must remain legible when the graphic
  is printed 8.5 inches wide. Avoid paragraphs inside diagrams.
- Wind encoding: use a viridis-like speed field with sparse white direction arrows.
  Never imply that color alone encodes direction.
- Composition: use the full canvas; no title above the illustration because the poster
  supplies the title. Include a small legend only when it conveys new information.
- Scientific restraint: diagrams must explain mechanisms, not make performance claims.

## D01 - Globe: bounded patches versus an unbounded coherent world

Create a wide landscape infographic, aspect ratio about 2.1:1. Center a simplified globe
with latitude/longitude grid lines. Wrap a continuous teal/blue wind ribbon with sparse
white vector arrows around the globe and allow it to continue past the visible horizon.
Place two small comparison insets on the left: (1) `ERA5 archive` represented by a stack
of dated rectangular maps, labeled `realistic, finite archive`; (2) `BLE-VAE` represented
by one 64-ish square regional patch with a hard boundary, labeled `learned, one bounded
patch`. On the right, show `ours` as several geographically separated query windows on
the same globe. Connect those windows with a subtle continuous wind band and one shared
seed icon, labeled `coherent, coordinate-indexed queries`. A dotted flight path from one
continent to another should cross multiple query windows without meeting a canvas edge.
Do not suggest the entire globe is generated eagerly; add a small cursor/crop symbol at
each requested region. The visual conclusion should be apparent without reading body
text: previous sources stop at an archive or patch; our method can query anywhere.

## D02 - Base model contract: conditioning, architecture, and a 4D sample

Create a wide pipeline diagram, aspect ratio about 2.2:1, with three clear stages.

Left, `inputs`: show Gaussian-noised wind maps as a stack, plus two clean coordinate
planes labeled `latitude` and `longitude`, plus a compact circular clock/calendar glyph
labeled `annual + semiannual + diurnal harmonics`, and a noise-scale badge `sigma`.

Center, `factorized space-time U-Net + EDM`: draw a compact U-Net silhouette with three
spatial resolutions. Use a repeated 2D grid icon for spatial residual blocks and a thin
horizontal time strip for 1D temporal convolutions. Include the concise annotation
`2D space per frame; 1D time between frames; joint denoising loss`. Do not draw vertical
convolution between atmospheric levels.

Right, `one generated sample`: make a matrix with columns `+0 h, +1 h, +2 h, +3 h` and
18 thin rows labeled from `level 49` at the bottom to `level 66` at the top. Enlarge three
representative rows so actual colored wind maps and arrows are visible; use brackets or
ellipsis to establish all 18 levels. Add the tensor label `4 x 36 x 64 x 64`, then decode
it underneath as `4 hours x (18 levels x u,v) x 64 x 64`. Add a small callout:
`height levels are separate channels; all are denoised jointly`.

## D03 - Qualitative ERA5 versus latest base model

Create an evidence-first comparison using actual exported data, not invented fields.
Aspect ratio about 2.1:1. Use two rows and four columns: top row `Held-out ERA5`, bottom
row `Base diffusion`; columns `+0 h`, `+1 h`, `+2 h`, `+3 h`. Choose one representative
held-out condition where the model captures a coherent evolving structure without using
the single worst case or cherry-picking the best. Use the same location, pressure/model
level, timestamps, crop, colormap limits, arrow density, and arrow scale in both rows.
Put each frame's maximum wind speed in a large high-contrast corner badge. Add one shared
horizontal color bar labeled `wind speed (m/s)`. Use short row labels outside the images
so they remain readable. Do not include BLE-VAE here; the purpose is to establish what
the bounded base model produces before explaining InfiniteDiffusion.

## D04 - InfiniteDiffusion dependency pyramid with cache reuse

Create a landscape explanatory diagram, aspect ratio about 1.45:1. The reading direction
must be top-to-bottom and match forward denoising: `high noise, k=K` at the top, then two
or three intermediate levels, then `clean wind, k=0` at the bottom. At every level draw
overlapping 64x64 windows as offset translucent squares. The requested output region
shrinks toward the bottom while its required context/parent region grows toward the top.

Use a consistent visual vocabulary tied to the poster pseudocode: outline requested
region `R` in orange; label local windows `Q_i`; show a small denoiser block `D_k` on
arrows between levels; show edge taper `w_i` as a transparent fade near window borders;
show coordinate and time conditioning `c_i` entering each denoiser from the side. At one
level, explicitly depict `weighted overlap -> normalize -> cache`.

Then add a second requested region `R2` offset to the right and partially overlapping
the first. Render the dependency branch already generated for `R1` in solid teal with a
small cache/database icon and the label `reuse cached state`; render only the newly
required branch in orange. Add a coordinate-hashed noise icon at the top labeled
`same seed + coordinate -> same noise`. Avoid equations; the mechanism should be clear
through arrows and labels alone.

## D05 - Corrected random-access zoom montage for a 64x64 base model

Create a three-panel horizontal zoom sequence, aspect ratio about 1.75:1, from the same
seed-defined virtual wind world. Panel 1 should show a multi-tile regional request, for
example `192 x 192 query = 3 x 3 base windows`. Draw a red 64x64 query box inside it.
Panel 2 should show exactly that red region enlarged and label it `64 x 64 query = one
base-model window`; draw a second smaller red sub-query box inside. Panel 3 should show
that sub-query enlarged and label it with its actual pixel dimensions.

Connect the top-right and bottom-right corners of each red box to the corresponding
top-left and bottom-left corners of the next panel using thin red diagonal lines. Make
the geometry unambiguous and prevent lines from crossing labels. Use the same wind-speed
colormap and arrow convention in every panel. Put a large readable top annotation across
the sequence: `same seed + same coordinates -> identical overlap`. Include a small note
`queries are generated from coordinates; this is not a crop from a stored global image`.
Do not repeat the obsolete claim that the base model is 192x192; 64x64 is the bounded
denoiser window.

## D06 - Single tile, naive tiling, and InfiniteDiffusion

Create a three-column controlled comparison, aspect ratio about 2.1:1, using the same
seed, conditioning, displayed region, and color scale.

Column A, `bounded base model`: one clean 64x64 wind patch with a visible hard outer edge.
Column B, `independent tiles + averaging`: a 3x3 mosaic of independently sampled 64x64
patches. Show thin internal tile boundaries; magnify one boundary in a circular inset so
the reader can see either a directional discontinuity or an over-smoothed overlap caused
by averaging. Label the failure `independent diffusion paths disagree`.
Column C, `InfiniteDiffusion`: the same 3x3 extent with internal boundaries visually
absent. Overlay faint dotted 64x64 window outlines only in the upper half so the reader
can tell it is tiled; show shared intermediate-state arrows crossing those outlines.
Magnify the same boundary location and label it `shared noise + cached overlap remains
coherent`.

At the bottom, add a compact comparison strip with three rows: `extent`, `seams`, and
`revisit consistency`. Values should read A: `64x64 / n.a. / yes`; B: `larger / blur or
seam / not guaranteed`; C: `unbounded / coordinated / yes`. Do not claim quantitative
superiority unless actual fields and benchmark outputs are used.

## D07 - Simplified PSD and spectral-residual explainer

Create a wide two-part scientific chart, aspect ratio about 1.4:1. Use actual evaluation
arrays if available. Keep only four curves: held-out `ERA5 reference` in charcoal,
`ERA5 self-split floor` in gray, `fitted simplex` in green, and `our base diffusion` in
navy or magenta. Omit white noise, phase shuffle, Helmholtz GP, unused checkpoints, and
any method not discussed in the poster.

Left two-thirds: log-log kinetic-energy power spectral density versus spatial frequency
or wavelength. Put wavelength in kilometers on a readable secondary x-axis if possible.
Shade the area between each method's log spectrum and ERA5 very lightly. Mark the scale
where our model begins to lose fine-scale energy with a vertical dashed line and a short
annotation `fine-scale power drops`.

Right one-third: explain the scalar spectral residual visually. Replot a small portion
of the ERA5 and model log spectra, draw three or four vertical double-headed arrows
between them, and label these arrows `log-spectrum differences`. Beneath, show the plain
language aggregation `spectral residual = average gap across compared scales`; then show
a horizontal mini-bar for the four methods with a downward arrow labeled `closer to the
ERA5 floor is better`. Use actual reported values where shown: floor 0.25, simplex 0.75,
ours 1.53. BLE-VAE may appear only as a fourth bar (2.73), not as another cluttering PSD
curve. Avoid dense legends and avoid more than one equation.

## Replacement checklist

1. Verify the generated artwork contains no fabricated numerical values.
2. Check every embedded label at the intended 8.5-inch print width.
3. Save source files and export PDF when the art is vector; otherwise export PNG at a
   minimum of 2400 pixels wide.
4. Replace only the matching `\artslot` in `poster.tex`; preserve its bounding box.
5. Rebuild with `make`, render the full poster, and inspect at both fit-to-page and 100%.
