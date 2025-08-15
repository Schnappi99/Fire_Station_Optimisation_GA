import geopandas as gpd

# 读取全国 LAD 边界
boundaries = gpd.read_file('../Data/boundary_json/Local_Authority_Districts_December_2024_Boundaries_UK_BFC_-8514277369542505193.geojson')

# 确保投影
if boundaries.crs.to_epsg() != 27700:
    boundaries = boundaries.to_crs(epsg=27700)

# 选取西米德兰兹城市
target_cities = [
    'Birmingham', 'Coventry', 'Wolverhampton',
    'Dudley', 'West Bromwich', 'Walsall', 'Sandwell', 'Solihull'
]
city_boundary = boundaries[boundaries['LAD21NM'].isin(target_cities)]

# 保存为 west_midlands.json
city_boundary.to_file('../Data/boundary_json/west_midlands.json', driver='GeoJSON')

print("Saved to ../Data/boundary_json/west_midlands.json")