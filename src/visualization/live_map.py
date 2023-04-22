import holoviews as hv
import holoviews.operation.datashader as hd
import bokeh as bk
import rioxarray as rxr
import numpy as np
import warnings
import os

# ## SET BOKEH BACKEND
hv.extension('bokeh', logo=False)

# ## SET CLASS MAPPING
color_key = {
    "Nan - Background": "#000000",
    "1 - Permanent grassland": "#a0db8e",
    "2 - Annual fruit and vegetable": "#cc5500",
    "3 - Summer cereals": "#e9de89",
    "4 - Winter cereals": "#f4ecb1",
    "5 - Rapeseed": "#dec928",
    "6 - Maize": "#f0a274",
    "7 - Annual forage crops": "#556b2f",
    "8 - Sugar beat": "#94861b",
    "9 - Flax and Hemp": "#767ee1",
    "10 - Permanent fruit": "#7d0015",
    "11 - Hopyards": "#9299a9",
    "12 - Vineyards": "#dea7b0",
    "13 - Other crops": "#ff0093",
    "14 - Not classified": "#f0f8ff",
}

# Visualization
legend = hv.NdOverlay(
    {
        k: hv.Points([0, 0], label=f"{k}").opts(color=v, size=0, apply_ranges=False)
        for k, v in color_key.items()
    },
    "Classification",
)

ticks = np.arange(len(color_key), dtype='float') + 0.0001
ticker = bk.models.FixedTicker(ticks=ticks)
labels = dict(zip(ticks, color_key))

# ## LOAD DATA
dataarray_mercator = rxr.open_rasterio('../../data/T33UWR_20190217.tif', masked=True).rio.reproject('EPSG:3857')

# ## SOME SETTINGS
# Some arbitrary sizes we will use to display images.
image_height, image_width = 600, 600

# Maps will have the same height, but they will be wider
map_height, map_width = image_height, 1000

# As we've seen, the coordinates in our dataset were called x and y, so we are
# going to use these.
key_dimensions = ['x', 'y']
value_dimension = dataarray_mercator.attrs["long_name"]  # 'Class'

clipping = {'NaN': '#00000000'}
hv.opts.defaults(
    hv.opts.Image(height=image_height, width=image_width, colorbar=True,
                  tools=['hover'], active_tools=['wheel_zoom'], clipping_colors=clipping,
                  alpha=0.8),
    hv.opts.Tiles(active_tools=['wheel_zoom'], height=map_height, width=map_width)
)

# ## LOAD BACKGROUND MAP FROM OSM
hv_tiles_osm = hv.element.tiles.OSM()

# ### CUSTOM TOOLTIP
# This is the JavaScript code that formats our coordinates:
# You only need to change the "digits" value if you would like to display the
# coordinates with more or fewer digits than 4.
formatter_code = """
  var digits = 4;
  var projections = Bokeh.require("core/util/projections");
  var x = special_vars.x;
  var y = special_vars.y;
  var coords = projections.wgs84_mercator.invert(x, y);
  return "" + (Math.round(coords[%d] * 10**digits) / 10**digits).toFixed(digits)+ "";
"""

# In the code above coords[%d] gives back the x coordinate if %d is 0, so at
# first we replace that.
formatter_code_x = formatter_code % 0
# Then we replace %d to 1 to get the y value.
formatter_code_y = formatter_code % 1

# This is the standard definition of a custom Holoviews Tooltip.
# Every line will be a line in the tooltip, with the first element being the
# label and the second element the displayed value.
custom_tooltips = [
    # We want to use @x and @y values, but send them to a custom formatter first.
    ('Lon', '@x{custom}'),
    ('Lat', '@y{custom}'),
    # This is where you should label and format your data:
    # '@image' is the data value of your GeoTIFF
    # In this example we format it as an integer and add "m" to the end of it
    # You can find more information on number formatting here:
    # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html#formatting-tooltip-fields
    ('Class', '@image{0}')
]

# For every value we marked above as {custom} we have to define what we mean by
# that. In this case for both variables we want to get a Bokeh CustomJSHover
# with respective JS codes created above.

custom_formatters = {
    '@x': bk.models.CustomJSHover(code=formatter_code_x),
    '@y': bk.models.CustomJSHover(code=formatter_code_y)
}

# We add these together, creating a custom Bokeh HoverTool
custom_hover = bk.models.HoverTool(tooltips=custom_tooltips, formatters=custom_formatters)

# ## DATASHADER INTEGRATION TO DYNAMICALLY DISPLAY HIGH-RESOLUTION IMAGES

hv_dataset_large = hv.Dataset(dataarray_mercator[0], vdims=value_dimension, kdims=key_dimensions)
hv_image_large = hv.Image(hv_dataset_large).opts(tools=[custom_hover],
                                                 cmap=tuple(color_key.values()),
                                                 clim=(-0.5, 14.5),
                                                 colorbar_opts={'ticker': ticker,
                                                                'major_label_overrides': labels})

hv_dyn_large = hd.regrid(hv_image_large)
hv_combined_large = hv_tiles_osm * hv_dyn_large

hv.save(hv_combined_large, '../../data/T33UWR_20190217.html')
