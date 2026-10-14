# Data, assets and third-party attributions

## Clinical dataset
**Extension of Z-Alizadeh Sani dataset** - UCI Machine Learning Repository, dataset #411.
Licence: CC BY 4.0. File: `data/raw/extention of Z-Alizadeh sani dataset.xlsx` (unmodified copy; the
training script can also re-download it from UCI).

> Alizadehsani, R., Roshanzamir, M., & Sani, Z. (2013). *extention of Z-Alizadeh sani dataset* [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5461K

> Alizadehsani, R. et al. (2013). A data mining approach for diagnosis of coronary artery disease.
> *Computer Methods and Programs in Biomedicine*, 111(1), 52-61.

## 3D anatomy
"BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International" (**CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/; licence updated by DBCLS on 2025-02-25).
Source: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/ (release 4.0, `partof_BP3D_4.0_obj_99.zip`)

`web/public/anatomy/heart.glb` is a derived work (meshes selected by FMA id, merged, decimated,
ventricular shells offset to the epicardium, per-vertex flow and territory attributes added by
`scripts/build_anatomy.py`). It is redistributed under CC BY 4.0 with the attribution above.

> Mitsuhashi, N., Fujieda, K., Tamura, T., Kawamoto, S., Takagi, T., & Okubo, K. (2009).
> BodyParts3D: 3D structure database for anatomical concepts. *Nucleic Acids Research*, 37, D782-D785.

## Clinical conventions used in the visualisation
- AHA 17-segment model and coronary territory assignment: Cerqueira, M. D. et al. (2002). Standardized
  myocardial segmentation and nomenclature for tomographic imaging of the heart. *Circulation*, 105(4), 539-542.
- Decision-curve analysis: Vickers, A. J., & Elkin, E. B. (2006). *Medical Decision Making*, 26(6), 565-574.
- Conformal prediction: Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*;
  Mondrian (class-conditional) variant.
- SHAP: Lundberg & Lee (2017), NeurIPS; TreeSHAP: Lundberg et al. (2020), *Nature Machine Intelligence*.

## Software and fonts
Three.js, React Three Fiber, drei (MIT); FastAPI, scikit-learn, LightGBM, SHAP (MIT/BSD);
ReportLab (BSD); IBM Plex Sans / Mono and Instrument Serif fonts (SIL Open Font License 1.1).

## LLM services (optional)
The copilot can call Groq-hosted `openai/gpt-oss-120b` (Apache-2.0 weights) and `whisper-large-v3-turbo`
(MIT weights). Only model outputs (probabilities, feature names and values) are sent; never names or
identifiers. The app runs fully offline without them.
