# Dataset attribution and license

## TinyHero dataset

- **Repository:** https://github.com/AgaMiko/pixel_character_generator (GitHub: AgaMiko)
- **File used:** `data.zip` (downloaded automatically by `dataset.py` / the Colab notebook; not redistributed in this project)
- **Contents (verified):** 3,648 PNG images, 64×64 RGBA, 912 characters × 4 viewing directions
  (`0` = back, `1` = left, `2` = front, `3` = right)
- **Origin of the art:** characters were assembled with the
  [Universal LPC Spritesheet Character Generator](https://github.com/makrohn/Universal-LPC-spritesheet/tree/7040e2fe85d2cb1e8154ec5fce382589d369bdb8)
  by makrohn, built on art from the Liberated Pixel Cup (LPC) contributors.

## License (as stated by the dataset repository)

> Under the rules of the LPC, all art submissions are dual licensed under GNU GPL 3.0 and CC-BY-SA 3.0.
> The work in this repository is licensed under the same terms.

- CC-BY-SA 3.0 — http://creativecommons.org/licenses/by-sa/3.0/ — full text: [`cc-by-sa-3.0.txt`](cc-by-sa-3.0.txt)
- GNU GPL 3.0 — http://www.gnu.org/licenses/gpl-3.0.html — full text: [`gpl-3.0.txt`](gpl-3.0.txt)

Both license files were copied unchanged from the dataset repository.

## What this means for PixelForge outputs

The generator is trained on CC-BY-SA / GPL art. If you publish trained weights, generated sprites or
derived images, credit the TinyHero dataset and the LPC artists as above and release them under
compatible share-alike terms. This note is provided for academic use and is not legal advice.
