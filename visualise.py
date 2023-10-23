import matplotlib.pyplot as plt  # plotting library
import matplotlib.patches as mpatches # draw domain boundry patches

# Issue: Maybe downgrade shapely pip install shapely==1.8.5
import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfeature # for coastlines


def plot_area(y_min, y_max, x_min, x_max):
    """Plot area in Antarctica w.r.t. on continental-scale map. 

    Args:
        y_min (_type_): Polar stereographic y min
        y_max (_type_): Polar stereographic
        x_min (_type_): Polar stereographic
        x_max (_type_): Polar stereographic
    """
    special_color = '#2D27EB'
    
    # Initialise plot
    fig = plt.figure(figsize = [6, 6])
    ax = plt.axes(projection = ccrs.SouthPolarStereo())

    # restrict to over 65 lat
    ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())

    # hides boundry line
    ax.axis('off')

    # add grey land
    ax.add_feature(cfeature.LAND, facecolor = ("#FAFAFA"), alpha = 1.0)

    # thin coastline lines
    ax.add_feature(cfeature.COASTLINE, edgecolor = special_color, linestyle = '-', linewidth = 0.4, alpha = 0.7)

    # patch
    ax.add_patch(mpatches.Rectangle(xy = [x_min, y_min], width = (x_max - x_min), height = (y_max - y_min),
                                facecolor = 'none', edgecolor = special_color, linewidth = 0.8,
                                transform = ccrs.SouthPolarStereo()))
    plt.show()