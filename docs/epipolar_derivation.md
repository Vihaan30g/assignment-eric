# Stereo depth from disparity

For a rectified stereo pair, corresponding image points lie on the same
horizontal epipolar line. Rectification makes epipolar lines horizontal, so
the correspondence search is one-dimensional.

```text
          left camera                         right camera
             C_L -------- baseline B -------- C_R
              |\                                |\
              | \                               | \
              |  \                              |  \
              |   \                             |   \
              +----\----------------------------+----\---- point P
                   x_L                         x_R
                   <----------- Z -------------->
```

Let the cameras have focal length `f` (in pixels), horizontal baseline `B`,
and the point be at depth `Z`. By similar triangles, its horizontal image
coordinates relative to each principal point are

```text
x_L = f X / Z
x_R = f (X - B) / Z
```

The disparity is `d = x_L - x_R = f B / Z`. Rearranging gives

```text
Z = f B / d
```

Use calibrated `f` in pixels, baseline in metres, and disparity in pixels;
the resulting depth is in metres. This assumes a rectified pair and positive,
valid disparity.
