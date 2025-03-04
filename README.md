# poorman_plot_digitizer

Poorman's plot digitizer in python, generated with Grok 3.

Features:
* Calibrate x-y axes and then start sampling.
* Zoom panel helps, with mouse-scrolling to adjust zoom.
* Press backspace to delete previous points (both in calibration and sampling modes).
* Save multiple data-sets with the same calibration.
* Data saved to workspace, to a combined .csv file, and to separate .txt files (optional).

---

## Use

Install the package locally using
```
pip install -e .
```
and then in python run
```
from poorman_plot_digitizer import poorman_plot_digitizer
image_path = "image_dir_path/example.png"
sampled_data = poorman_plot_digitizer.main(image_path, save_csv=True, save_txt=True)
```
or run inline
```
python package_dir_path/poorman_plot_digitizer/poorman_plot_digitizer.py
```