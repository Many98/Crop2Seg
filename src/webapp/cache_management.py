import os
import shutil
import time
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_size(dir_path: str):
    size = 0
    try:
        for path, dirs, files in os.walk(dir_path):
            for f in files:
                fp = os.path.join(path, f)
                size += os.path.getsize(fp)
    except:
        return 0
    return size / 1e9


def cache_mgmt():
    st.header('Cache management', divider='rainbow')
    st.caption('Manage disk usage of cached files')

    st.info(f"Cache storage is located at {os.path.join(os.getcwd(), 'src', 'webapp', 'cache')}", icon="ℹ️")

    st.info(f"Exported rasters are located at {os.path.join(os.getcwd(), 'data', 'export')}", icon="ℹ️")

    names = ['lpis', 'prediction', 's2_patches', 's2_tiles']
    sizes = [get_size(os.path.join(os.getcwd(), 'src', 'webapp', 'cache', i)) for i in names] + [
        get_size(os.path.join(os.getcwd(), 'data', 'export'))]
    labels = np.array(['LPIS', 'Prediction', 'S2 Patches', 'S2 Tiles', 'Rasters'])

    col1, col2 = st.columns([0.15, 0.85])

    with col1:
        delete_lpis = st.toggle('Delete LPIS', value=True and sizes[0] >= 0.00001,
                                help='Delete LPIS cached data',
                                disabled=sizes[0] < 0.00001
                                )
        delete_prediction = st.toggle('Delete prediction', value=False,
                                      help='Delete cached prediction data',
                                      disabled=sizes[1] < 0.00001
                                      )
        delete_patches = st.toggle('Delete S2 patches', value=True and sizes[2] >= 0.00001,
                                   help='Delete cached time series data',
                                   disabled=sizes[2] < 0.00001
                                   )
        delete_tiles = st.toggle('Delete S2 tiles', value=False,
                                 help='Delete cached Sentinel-2 tiles',
                                 disabled=sizes[3] < 0.00001
                                 )

        delete_rasters = st.toggle('Delete rasters', value=False,
                                   help='Delete exported predictions in tif format',
                                   disabled=sizes[4] < 0.00001
                                   )

        remove_cache = st.button('Delete',
                                 help=f"Delete chosen cached files")

        if remove_cache:
            try:
                if delete_lpis:
                    shutil.rmtree('src/webapp/cache/lpis')

                if delete_prediction:
                    shutil.rmtree('src/webapp/cache/prediction')

                if delete_patches:
                    shutil.rmtree('src/webapp/cache/s2_patches')

                if delete_tiles:
                    shutil.rmtree('src/webapp/cache/s2_tiles')

                if delete_rasters:
                    shutil.rmtree('data/export')

                st.success("Data deleted successfully.", icon="✅")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(
                    f"Error occured when deleting cache: {e}")


    with col2:

        fig, ax = plt.subplots(figsize=(7, 3))

        x = np.arange(len(labels))
        width = 0.25
        for p, s, l in zip(x, sizes, labels):
            ax.bar(p, s, width, label=l)
            ax.text(p-0.1, s+0.3, f'{round(s, 3)} GB', fontsize=4)

        ax.set_xticks(x, labels)
        ax.set_ylabel("Size in GB", fontsize=6)
        ax.set_title('Cache disk usage', fontsize=6)
        ax.legend(loc='upper left', ncols=3, fontsize=4)

        st.pyplot(fig, use_container_width=False)


