import os

import pandas as pd
import pytest

import jimg_ncd.config as cfg
from jimg_ncd.nuclei import GroupAnalysis, NucleiDataManagement, NucleiFinder, test_data


@pytest.fixture(autouse=True)
def disable_display():
    cfg._DISPLAY_MODE = False


def test_load_test_data():
    test_data()
    assert os.path.exists("test_data")


@pytest.fixture
def nf_microscope():
    nf = NucleiFinder()
    image = nf.load_image("test_data/microscope_nuclei/r01c02f90p20-ch1sk1fk1fl1.tiff")
    nf.input_image(image)
    return nf


def test_basic_parameters(nf_microscope):
    nf = nf_microscope
    params = nf.current_parameters_nuclei
    assert params is not None


def test_nuclei_segmentation_and_selection(nf_microscope):
    nf = nf_microscope
    nf.set_nuclei_circularity(0.5)
    nf.set_nuclei_size((100, 800))
    nf.set_nuclei_min_mean_intensity(2000)
    nf.find_nuclei()
    nf.select_nuclei()
    results, img = nf.get_results_nuclei_selected()
    assert len(results) > 0


def test_chromatinization(nf_microscope):
    nf = nf_microscope
    nf.find_nuclei()
    nf.set_chromatinization_size((2, 400))
    nf.set_chromatinization_ratio(0.05)
    nf.set_chromatinization_cut_point(0.95)
    nf.nuclei_chromatinization()
    results, img = nf.get_results_nuclei_chromatinization()
    assert len(results) > 0


###############################################################################


@pytest.fixture
def nf():
    nf = NucleiFinder()
    image = nf.load_image("test_data/flow_cytometry/ctrl/3087_Ch7.ome.tif")
    nf.input_image(image)
    return nf


def test_nuclei_finder(nf):
    nf.nuclei_finder_test()
    nf.set_nms(0.6)
    nf.set_prob(0.3)
    nf.find_nuclei()
    results, img = nf.get_results_nuclei()
    assert len(results) > 0


def test_nuclei_selection(nf):
    nf.find_nuclei()
    nf.select_nuclei()
    nf.set_nuclei_circularity(0.5)
    nf.set_nuclei_size((100, 800))
    nf.set_nuclei_min_mean_intensity(2000)
    nf.select_nuclei()
    results, img = nf.get_results_nuclei_selected()
    assert len(results) > 0


def test_chromatinization(nf):
    nf.find_nuclei()
    nf.nuclei_chromatinization()
    nf.set_chromatinization_size((2, 1000))
    nf.set_chromatinization_ratio(0.005)
    nf.set_chromatinization_cut_point(1.05)
    nf.set_adj_chrom_gamma(0.25)
    nf.set_adj_chrom_contrast(4)
    nf.set_adj_chrom_brightness(950)
    nf.nuclei_chromatinization()
    results, img = nf.get_results_nuclei_chromatinization()
    assert len(results) > 0


@pytest.fixture
def ndm():
    nf = NucleiFinder()
    series_results_ctrl = nf.series_analysis_chromatinization(
        path_to_images="test_data/flow_cytometry/ctrl",
        file_extension="tif",
        selected_id=[],
        selection_opt=True,
        include_img=False,
        test_series=0,
    )
    series_results_dis = nf.series_analysis_chromatinization(
        path_to_images="test_data/flow_cytometry/dis",
        file_extension="tif",
        selected_id=[],
        selection_opt=True,
        include_img=False,
        test_series=0,
    )
    ndm_ctrl = NucleiDataManagement(series_results_ctrl, "healthy_chromatinization")
    ndm_dis = NucleiDataManagement(series_results_dis, "disease_chromatinization")
    return ndm_ctrl, ndm_dis


def test_ndm_data(ndm):
    ndm_ctrl, ndm_dis = ndm
    df_ctrl = ndm_ctrl.get_data()
    df_dis = ndm_dis.get_data()
    assert not df_ctrl.empty
    assert not df_dis.empty


def test_add_IS_data(ndm):
    ndm_ctrl, ndm_dis = ndm
    healthy_data = pd.read_csv("test_data/flow_cytometry/ctrl.txt", sep="\t", header=1)
    disease_data = pd.read_csv("test_data/flow_cytometry/dis.txt", sep="\t", header=1)
    selected_columns = [
        "Area_M01",
        "Major Axis_M01",
        "Minor Axis_M01",
        "Aspect Ratio_M01",
        "Diameter_M01",
        "Area_M09",
        "Major Axis_M09",
        "Minor Axis_M09",
        "Aspect Ratio_M09",
        "Diameter_M09",
    ]
    ndm_ctrl.add_IS_data(healthy_data, IS_features=selected_columns)
    ndm_dis.add_IS_data(disease_data, IS_features=selected_columns)
    df_is_ctrl = ndm_ctrl.get_data_with_IS()
    df_is_dis = ndm_dis.get_data_with_IS()
    assert not df_is_ctrl.empty
    assert not df_is_dis.empty


def test_concat_data(ndm):
    ndm_ctrl, ndm_dis = ndm
    healthy_data = pd.read_csv("test_data/flow_cytometry/ctrl.txt", sep="\t", header=1)
    disease_data = pd.read_csv("test_data/flow_cytometry/dis.txt", sep="\t", header=1)
    selected_columns = [
        "Area_M01",
        "Major Axis_M01",
        "Minor Axis_M01",
        "Aspect Ratio_M01",
        "Diameter_M01",
        "Area_M09",
        "Major Axis_M09",
        "Minor Axis_M09",
        "Aspect Ratio_M09",
        "Diameter_M09",
    ]
    ndm_ctrl.add_IS_data(healthy_data, IS_features=selected_columns)
    ndm_dis.add_IS_data(disease_data, IS_features=selected_columns)
    ndm_ctrl.add_experiment([ndm_dis])

    df = ndm_ctrl.get_mutual_experiments_data(inc_is=True)
    ndm_ctrl.save_mutual_experiments(path="", inc_is=True)

    assert not df.empty


def test_group_analysis():
    data = pd.read_csv(
        "healthy_chromatinization_disease_chromatinization_IS.csv", sep=",", header=0
    )
    ga = GroupAnalysis.load_data(data, ids_col="id_name", set_col="set")
    ga.DFA(
        meta_group_by="sets",
        sets={
            "disease": ["disease_chromatinization"],
            "ctrl": ["healthy_chromatinization"],
        },
        n_proc=1,
    )
    dfa_results = ga.get_DFA()
    assert not dfa_results.empty
    ga.data_scale()
    ga.PCA()
    ga.UMAP(
        PC_num=8,
        factorize_with_metadata=False,
        harmonize_sets=True,
        n_neighbors=10,
        min_dist=0.01,
        n_components=2,
    )
    ga.db_scan(eps=0.5, min_samples=10)
    ga.UMAP_on_clusters(min_entities=5)
    full_data = ga.full_info()
    assert not full_data.empty


def test_proportion_analysis():

    data = pd.read_csv(
        "healthy_chromatinization_disease_chromatinization_IS.csv", sep=",", header=0
    )
    ga = GroupAnalysis.load_data(data, ids_col="id_name", set_col="set")

    ga.proportion_analysis(grouping_col="sets", val_col="nuclei_per_img")

    pl = ga.get_proportion_plot(show=False)
    assert pl is not None
    stats = ga.get_proportion_stats()
    assert stats is not None
    assert not stats.empty
