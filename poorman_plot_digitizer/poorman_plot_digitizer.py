import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import tkinter as tk
import os

# Global variable to store sampled data
sampled_data = {}

def get_unique_filename(base_name, extension):
    """Generate a unique filename by appending _vN if file exists."""
    counter = 2
    new_name = base_name
    while os.path.exists(f"{new_name}.csv") or any(os.path.exists(f"{new_name}_{i}.txt") for i in range(1, counter)):
        new_name = f"{base_name}_v{counter}"
        counter += 1
    return new_name

def load_and_display_image(image_path):
    """Load and display the image with a zoom panel, maximizing figure size."""
    img = mpimg.imread(image_path)
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    dpi = plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(screen_width / dpi * 0.95, screen_height / dpi * 0.95))
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except AttributeError:
        try:
            manager.window.state('zoomed')
        except AttributeError:
            pass

    ax = fig.add_axes([0.05, 0.05, 0.55, 0.9])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    zoom_ax = fig.add_axes([0.65, 0.25, 0.3, 0.5])
    zoom_ax.set_visible(False)
    zoom_size = 150
    return fig, ax, zoom_ax, img, zoom_size

def pixel_to_data(pixel_x, pixel_y, calib_values, pixel_calib, img_shape, x_scale, y_scale):
    """Convert pixel coordinates to calibrated data coordinates based on axis scales."""
    x_min, x_max, y_min, y_max = calib_values
    x1, x2, y1_pix, y2_pix = pixel_calib

    # Calculate mapped values (actual for linear, exponent for logarithmic)
    mapped_x = x_min + (pixel_x - x1) * (x_max - x_min) / (x2 - x1)
    mapped_y = y_min - (pixel_y - y1_pix) * (y_max - y_min) / (y1_pix - y2_pix)

    x = mapped_x if x_scale == 'linear' else 10 ** mapped_x
    y = mapped_y if y_scale == 'linear' else 10 ** mapped_y

    return x, y

def save_set_to_txt(dataset, filename):
    """Save a single dataset to a TXT file with two columns (x, y) using calibrated coordinates."""
    with open(filename, 'w') as f:
        for x, y in dataset[0]:
            f.write(f"{x} {y}\n")
    print(f"Saved {filename}")

def digitize_figure(fig, ax, zoom_ax, img, zoom_size, save_csv=True, save_txt=True, file_name="dataset"):
    """Handle calibration and sampling with undo and fine adjustments."""
    global sampled_data

    # Get unique filename if files exist
    file_name = get_unique_filename(file_name, ".csv")

    # Initialize variables, starting with scale selection
    calib_state = -2  # Start with x-axis scale question
    x_scale = None
    y_scale = None
    scale_input = ''
    calib_points = []
    calib_values = [None, None, None, None]
    calib_markers = []
    current_value = []
    all_data = {}
    dataset_num = 1
    current_points = []
    markers = []
    sampling_mode = False
    color_list = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "orange", "purple", "brown", "pink", "gray"]
    marker_list = ['o', 's', '^', 'v', 'D', '*']
    move_step = 1

    text = ax.text(0.05, 0.95, "Is the X-axis linear (0) or logarithmic (1)? Press 0 or 1, then Enter.",
                   transform=ax.transAxes, color='white', bbox=dict(facecolor='black', alpha=0.8))

    def update_zoom_panel(x, y):
        """Update the zoom panel centered on the given coordinates."""
        nonlocal zoom_size
        half_size = zoom_size // 2

        x_min = max(0, x - half_size)
        x_max = min(img.shape[1], x + half_size)
        if x_min == 0:
            x_max = min(img.shape[1], zoom_size)
        elif x_max == img.shape[1]:
            x_min = max(0, img.shape[1] - zoom_size)

        y_min = max(0, y - half_size)
        y_max = min(img.shape[0], y + half_size)
        if y_min == 0:
            y_max = min(img.shape[0], zoom_size)
        elif y_max == img.shape[0]:
            y_min = max(0, img.shape[0] - zoom_size)

        if x_max <= x_min or y_max <= y_min:
            zoom_ax.set_visible(False)
        else:
            zoom_ax.set_visible(True)
            zoom_ax.clear()
            zoom_ax.imshow(img[y_min:y_max, x_min:x_max])
            zoom_x_center = x - x_min
            zoom_y_center = y - y_min
            zoom_ax.axvline(x=zoom_x_center, color='grey', linewidth=0.5)
            zoom_ax.axhline(y=zoom_y_center, color='grey', linewidth=0.5)

            for ds_name, (_, pixel_points) in all_data.items():
                ds_idx = int(ds_name.split('_')[1]) - 1
                color_idx = ds_idx % len(color_list)
                marker_idx = ds_idx // len(color_list) % len(marker_list)
                color = color_list[color_idx]
                marker = marker_list[marker_idx]
                for px, py in pixel_points:
                    if x_min - 0.5 <= px <= x_max + 0.5 and y_min - 0.5 <= py <= y_max + 0.5:
                        zoom_ax.plot(px - x_min, py - y_min, color=color, marker=marker, markersize=5,
                                     markeredgecolor='black', markeredgewidth=1)

            if current_points and sampling_mode:
                color_idx = (dataset_num - 1) % len(color_list)
                marker_idx = (dataset_num - 1) // len(color_list) % len(marker_list)
                color = color_list[color_idx]
                marker = marker_list[marker_idx]
                for px, py in current_points:
                    if x_min - 0.5 <= px <= x_max + 0.5 and y_min - 0.5 <= py <= y_max + 0.5:
                        zoom_ax.plot(px - x_min, py - y_min, color=color, marker=marker, markersize=5,
                                     markeredgecolor='black', markeredgewidth=1)

            for i, (px, py) in enumerate(calib_points):
                if x_min - 0.5 <= px <= x_max + 0.5 and y_min - 0.5 <= py <= y_max + 0.5:
                    color = 'blue' if i < 2 else 'red'
                    zoom_ax.plot(px - x_min, py - y_min, color=color, marker='*', markersize=10,
                                 markeredgecolor='black', markeredgewidth=1)

            zoom_ax.axis('off')
        fig.canvas.draw_idle()

    def on_mouse_move(event):
        if event.inaxes == ax and event.xdata and event.ydata:
            update_zoom_panel(int(event.xdata), int(event.ydata))

    def on_scroll(event):
        nonlocal zoom_size
        if event.inaxes == ax:
            zoom_size = max(50, zoom_size - 10) if event.button == 'up' else min(300, zoom_size + 10)
            if event.xdata and event.ydata:
                update_zoom_panel(int(event.xdata), int(event.ydata))

    def on_click(event):
        nonlocal calib_state, calib_points, calib_markers, current_points, markers, sampling_mode
        if event.inaxes != ax or event.button != 1 or calib_state < 0:
            return
        if not sampling_mode and calib_state in [0, 2, 4, 6]:
            px, py = event.xdata, event.ydata
            calib_points.append((px, py))
            marker = ax.plot(px, py, 'b*' if calib_state < 4 else 'r*', markersize=10, markeredgecolor='black',
                             markeredgewidth=1)[0]
            calib_markers.append(marker)
            if calib_state == 0:
                text.set_text("Enter value for X-axis point 1: ")
            elif calib_state == 2:
                text.set_text("Enter value for X-axis point 2: ")
            elif calib_state == 4:
                text.set_text("Enter value for Y-axis point 1: ")
            elif calib_state == 6:
                text.set_text("Enter value for Y-axis point 2: ")
            calib_state += 1
            update_zoom_panel(int(px), int(py))
        elif sampling_mode:
            px, py = event.xdata, event.ydata
            current_points.append((px, py))
            color_idx = (dataset_num - 1) % len(color_list)
            marker_idx = (dataset_num - 1) // len(color_list) % len(marker_list)
            marker_plot = ax.plot(px, py, color=color_list[color_idx], marker=marker_list[marker_idx], markersize=5,
                                  markeredgecolor='black', markeredgewidth=1)[0]
            markers.append(marker_plot)
            update_zoom_panel(int(px), int(py))

    def on_key(event):
        nonlocal calib_state, x_scale, y_scale, scale_input, calib_points, calib_markers, current_value, calib_values, sampling_mode, current_points, markers, dataset_num, all_data

        # Handle x-axis scale input
        if calib_state == -2:
            if event.key in ['0', '1']:
                scale_input = event.key
                text.set_text(f"Is the X-axis linear (0) or logarithmic (1)? Current: {scale_input}")
            elif event.key == 'backspace':
                scale_input = ''
                text.set_text("Is the X-axis linear (0) or logarithmic (1)? ")
            elif event.key == 'enter':
                if scale_input == '0':
                    x_scale = 'linear'
                    calib_state = -1
                    scale_input = ''
                    text.set_text("Is the Y-axis linear (0) or logarithmic (1)? Press 0 or 1, then Enter.")
                elif scale_input == '1':
                    x_scale = 'logarithmic'
                    calib_state = -1
                    scale_input = ''
                    text.set_text("Is the Y-axis linear (0) or logarithmic (1)? Press 0 or 1, then Enter.")
                else:
                    text.set_text("Invalid input. Press 0 or 1, then Enter.")
                    scale_input = ''
            fig.canvas.draw_idle()
            return

        # Handle y-axis scale input
        if calib_state == -1:
            if event.key in ['0', '1']:
                scale_input = event.key
                text.set_text(f"Is the Y-axis linear (0) or logarithmic (1)? Current: {scale_input}")
            elif event.key == 'backspace':
                scale_input = ''
                text.set_text("Is the Y-axis linear (0) or logarithmic (1)? ")
            elif event.key == 'enter':
                if scale_input == '0':
                    y_scale = 'linear'
                    calib_state = 0
                    scale_input = ''
                    text.set_text("Click X-axis point 1:")
                elif scale_input == '1':
                    y_scale = 'logarithmic'
                    calib_state = 0
                    scale_input = ''
                    text.set_text("Click X-axis point 1:")
                else:
                    text.set_text("Invalid input. Press 0 or 1, then Enter.")
                    scale_input = ''
            fig.canvas.draw_idle()
            return

        # Calibration phase
        if not sampling_mode and calib_state < 8:
            if calib_state in [1, 3, 5, 7]:
                if event.key.isdigit() or event.key in ['.', '-']:
                    current_value.append(event.key)
                    text.set_text(text.get_text() + event.key)
                elif event.key == 'enter' and current_value:
                    calib_values[calib_state // 2] = float(''.join(current_value))
                    current_value = []
                    if calib_state == 7:
                        sampling_mode = True
                        text.set_text(f"Sampling Data Set {dataset_num}: Click to sample, Backspace to undo, Enter to finish.")
                        fig.text(0.65, 0.80,
                                 f"Calibration: Xmin={calib_values[0]}, Xmax={calib_values[1]}, Ymin={calib_values[2]}, Ymax={calib_values[3]}",
                                 color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.8))
                    else:
                        calib_state += 1
                        if calib_state < 4:
                            text.set_text(f"Click X-axis point {(calib_state // 2) + 1}:")
                        else:
                            text.set_text(f"Click Y-axis point {((calib_state - 4) // 2) + 1}:")
                elif event.key == 'backspace':
                    if current_value:
                        current_value.pop()
                        text.set_text(text.get_text()[:-1])
                    elif calib_points:
                        calib_points.pop()
                        calib_markers.pop().remove()
                        calib_values[calib_state // 2] = None
                        calib_state = max(0, calib_state - 2)
                        if calib_state < 4:
                            text.set_text(f"Click X-axis point {(calib_state // 2) + 1}:")
                        else:
                            text.set_text(f"Click Y-axis point {((calib_state - 4) // 2) + 1}:")
                elif event.key in ['up', 'down', 'left', 'right'] and calib_points:
                    px, py = calib_points[-1]
                    if event.key == 'up':
                        py = max(0, py - move_step)
                    elif event.key == 'down':
                        py = min(img.shape[0] - 1, py + move_step)
                    elif event.key == 'left':
                        px = max(0, px - move_step)
                    elif event.key == 'right':
                        px = min(img.shape[1] - 1, px + move_step)
                    calib_points[-1] = (px, py)
                    calib_markers[-1].set_data([px], [py])
                    update_zoom_panel(int(px), int(py))
                fig.canvas.draw_idle()
            elif event.key == 'backspace' and calib_points:
                calib_points.pop()
                calib_markers.pop().remove()
                calib_state = max(0, calib_state - 2)
                if calib_state < 4:
                    text.set_text(f"Click X-axis point {(calib_state // 2) + 1}:")
                else:
                    text.set_text(f"Click Y-axis point {((calib_state - 4) // 2) + 1}:")
                fig.canvas.draw_idle()
                if calib_points:
                    update_zoom_panel(int(calib_points[-1][0]), int(calib_points[-1][1]))
            elif event.key in ['up', 'down', 'left', 'right'] and calib_points:
                px, py = calib_points[-1]
                if event.key == 'up':
                    py = max(0, py - move_step)
                elif event.key == 'down':
                    py = min(img.shape[0] - 1, py + move_step)
                elif event.key == 'left':
                    px = max(0, px - move_step)
                elif event.key == 'right':
                    px = min(img.shape[1] - 1, px + move_step)
                calib_points[-1] = (px, py)
                calib_markers[-1].set_data([px], [py])
                update_zoom_panel(int(px), int(py))
                fig.canvas.draw_idle()

        # Sampling phase
        elif sampling_mode:
            if event.key == 'backspace' and current_points:
                current_points.pop()
                markers.pop().remove()
                fig.canvas.draw_idle()
                if current_points:
                    update_zoom_panel(int(current_points[-1][0]), int(current_points[-1][1]))
            elif event.key == 'enter' and current_points:
                calibrated_points = np.array([pixel_to_data(px, py, calib_values,
                                                            [calib_points[0][0], calib_points[1][0],
                                                             calib_points[2][1], calib_points[3][1]],
                                                            img.shape, x_scale, y_scale)
                                              for px, py in current_points])
                all_data[f"dataset_{dataset_num}"] = (calibrated_points, current_points.copy())
                if save_txt:
                    save_set_to_txt(all_data[f"dataset_{dataset_num}"], f"{file_name}_{dataset_num}.txt")
                dataset_num += 1
                current_points = []
                markers = []
                text.set_text(f"Sampling Data Set {dataset_num}: Click to sample, Backspace to undo, Enter to finish.")
                sampled_data.update({k: v[0] for k, v in all_data.items()})
                if save_csv:
                    save_to_csv({k: v[0] for k, v in all_data.items()}, f"{file_name}.csv")
                fig.canvas.draw_idle()
            elif event.key in ['up', 'down', 'left', 'right'] and current_points:
                px, py = current_points[-1]
                if event.key == 'up':
                    py = max(0, py - move_step)
                elif event.key == 'down':
                    py = min(img.shape[0] - 1, py + move_step)
                elif event.key == 'left':
                    px = max(0, px - move_step)
                elif event.key == 'right':
                    px = min(img.shape[1] - 1, px + move_step)
                current_points[-1] = (px, py)
                markers[-1].set_data([px], [py])
                update_zoom_panel(int(px), int(py))
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    return all_data

def save_to_csv(all_data, filename="dataset.csv"):
    """Save all data to a single CSV file using calibrated coordinates."""
    if not all_data:
        return
    df_dict = {}
    max_len = max(len(points) for points in all_data.values())
    for ds_name, points in all_data.items():
        x_vals = [p[0] for p in points] + [np.nan] * (max_len - len(points))
        y_vals = [p[1] for p in points] + [np.nan] * (max_len - len(points))
        df_dict[f"{ds_name}_x"] = x_vals
        df_dict[f"{ds_name}_y"] = y_vals
    df = pd.DataFrame(df_dict)
    df.to_csv(filename, index=False)
    print(f"All data saved to {filename}")

def main(image_path=None, save_csv=True, save_txt=True, file_name="dataset"):
    """Main function with optional parameters."""
    global sampled_data
    if image_path is None:
        image_path = input("Please enter the path to the image file: ")
    fig, ax, zoom_ax, img, zoom_size = load_and_display_image(image_path)
    all_data = digitize_figure(fig, ax, zoom_ax, img, zoom_size, save_csv, save_txt, file_name)
    sampled_data.update({k: v[0] for k, v in all_data.items()})
    plt.show()
    return sampled_data

if __name__ == "__main__":
    sampled_data = main(save_csv=True, save_txt=True)