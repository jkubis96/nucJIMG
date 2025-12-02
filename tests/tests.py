from jimg_ncd.nuclei import test_data

test_data()


from jimg_ncd.nuclei import NucleiFinder

# initiate class
nf = NucleiFinder()


image = nf.load_image("test_data/microscope_nuclei/r01c02f90p20-ch1sk1fk1fl1.tiff")


nf.input_image(image)


# Check the basic parameters
nf.current_parameters_nuclei


# Test nms & prob parmeters for nuclei segmentation
nf.nuclei_finder_test()

nf.browser_test()

###############################################################################

# If required, change parameters
nf.set_nms(nms=0.9)

nf.set_prob(prob=0.5)


# Analysis

# 1. First step on nuclei analysis
nf.find_nuclei()


# Parameters for micrsocope image adjustment
nf.current_parameters_img_adj

###############################################################################


# If image required changes, change parameters and run again (nf.find_nuclei())
nf.set_adj_image_brightness(brightness=1000)

nf.set_adj_image_gamma(gamma=1.2)

nf.set_adj_image_contrast(contrast=2)


# Check if parameters has changed
nf.current_parameters_nuclei


# Second execution with new parameters for image adjustment
nf.find_nuclei()

###############################################################################

# Return results
nuclei_results, analysed_img = nf.get_results_nuclei()

###############################################################################


# 2. Second step of analysis (selection)


nf.select_nuclei()

###############################################################################

# Parameters for selecting nuclei; adjust if analysis results do not meet
# requirements, and re-run the analysis as needed.
nf.current_parameters_nuclei

nf.set_nuclei_circularity(circ=0.5)

nf.set_nuclei_size(size=(100, 800))

nf.set_nuclei_min_mean_intensity(intensity=2000)


# Check if parameters has changed
nf.current_parameters_nuclei


# Second execution with adjusted parameters of second step of analysis (selection)
nf.select_nuclei()


###############################################################################


# Return results
nuclei_selected_results, analysed_selected_img = nf.get_results_nuclei_selected()


###############################################################################

# 3. third step (chromatinization alaysis)
nf.nuclei_chromatinization()

###############################################################################

# Parameters for nuclei chromatinization; adjust if analysis results do not meet
# requirements, and re-run the analysis as needed.


# Chromatinization parameters

nf.current_parameters_chromatinization

nf.set_chromatinization_size(size=(2, 400))

nf.set_chromatinization_ratio(ratio=0.05)

nf.set_chromatinization_cut_point(cut_point=0.95)

nf.current_parameters_chromatinization

# Chromatinization image parameters

nf.current_parameters_img_adj_chro

nf.set_adj_chrom_gamma(gamma=0.25)

nf.set_adj_chrom_contrast(contrast=3)

nf.set_adj_chrom_brightness(brightness=950)

nf.current_parameters_img_adj_chro


# Second execution of the third step (chromatinization analysis)
nf.nuclei_chromatinization()

###############################################################################

chromatinization_results, analysed_chromatinization_img = (
    nf.get_results_nuclei_chromatinization()
)


###############################################################################

# If your parameters are correct for your data, you can run series analysis on more images

# Nuclei

series_results_nuclei = nf.series_analysis_nuclei(
    path_to_images="test_data/microscope_nuclei",
    file_extension="tiff",
    selected_id=[],
    fille_name_part="ch1",
    selection_opt=True,
    include_img=False,
    test_series=0,
)

###############################################################################

# get & save results

from jimg_ncd.nuclei import NucleiDataManagement

# initiate class with NucleiFinder data
ndm = NucleiDataManagement(series_results_nuclei, "example")

# get data as data frame
df = ndm.get_data()
print(df)

# save results as data frame
ndm.save_results_df(path="")

# save results as project *.nuc
ndm.save_nuc_project(path="")

# saved project by ndm.save_nuc_project(path='') can be then loaded for further analysis
ndm2 = NucleiDataManagement.load_nuc_dict("example.nuc")
ndm2.get_data()


# check
all(
    (v.equals(ndm2.__dict__[k]) if hasattr(v, "equals") else v == ndm2.__dict__[k])
    for k, v in ndm.__dict__.items()
)


###############################################################################

# Chromatinization

series_results_chromatinization = nf.series_analysis_chromatinization(
    path_to_images="test_data/microscope_nuclei",
    file_extension="tiff",
    selected_id=[],
    fille_name_part="ch1",
    selection_opt=True,
    include_img=True,
    test_series=0,
)

###############################################################################

# get & save results

from jimg_ncd.nuclei import NucleiDataManagement

# initiate class with NucleiFinder data
ndm = NucleiDataManagement(series_results_chromatinization, "example_chromatinization")

# get data as data frame
df = ndm.get_data()
print(df)

# save results as data frame
ndm.save_results_df(path="")

# save results as project *.nuc
ndm.save_nuc_project(path="")

# saved project by ndm.save_nuc_project(path='') can be then loaded for further analysis
ndm2 = NucleiDataManagement.load_nuc_dict("example_chromatinization.nuc")
ndm2.get_data()


# check
all(
    (v.equals(ndm2.__dict__[k]) if hasattr(v, "equals") else v == ndm2.__dict__[k])
    for k, v in ndm.__dict__.items()
)


###############################################################################

# flow cytometry

from jimg_ncd.nuclei import NucleiFinder

# initiate class
nf = NucleiFinder()


image = nf.load_image("test_data/flow_cytometry/ctrl/3087_Ch7.ome.tif")


nf.input_image(image)


# Check the basic parameters
nf.current_parameters_nuclei


# Test nms & prob parmeters for nuclei segmentation
nf.nuclei_finder_test()

nf.browser_test()


###############################################################################

# If required, change parameters
nf.set_nms(nms=0.6)

nf.set_prob(prob=0.3)


# Analysis

# 1. First step on nuclei analysis
nf.find_nuclei()

###############################################################################


# Parameters for micrsocope image adjustment
nf.current_parameters_img_adj


# If image required changes, change parameters and run again (nf.find_nuclei())
nf.set_adj_image_brightness(brightness=1000)

nf.set_adj_image_gamma(gamma=1.2)

nf.set_adj_image_contrast(contrast=2)


# Check if parameters has changed
nf.current_parameters_nuclei


# Second execution with new parameters for image adjustment
nf.find_nuclei()


###############################################################################


# Return results
nuclei_results, analysed_img = nf.get_results_nuclei()

###############################################################################

# 2. Second step of analysis (selection)
nf.select_nuclei()


###############################################################################

# Parameters for selecting nuclei; adjust if analysis results do not meet
# requirements, and re-run the analysis as needed.
nf.current_parameters_nuclei

nf.set_nuclei_circularity(circ=0.5)


nf.set_nuclei_size(size=(100, 800))

nf.set_nuclei_min_mean_intensity(intensity=2000)


# Check if parameters has changed
nf.current_parameters_nuclei


# Second execution with adjusted parameters of second step of analysis (selection)
nf.select_nuclei()

###############################################################################

# Return results
nuclei_selected_results, analysed_selected_img = nf.get_results_nuclei_selected()

###############################################################################

# 3. third step (chromatinization alaysis)
nf.nuclei_chromatinization()

###############################################################################

# Parameters for nuclei chromatinization; adjust if analysis results do not meet
# requirements, and re-run the analysis as needed.


# Chromatinization parameters
nf.current_parameters_chromatinization

nf.set_chromatinization_size(size=(2, 1000))

nf.set_chromatinization_ratio(ratio=0.005)

nf.set_chromatinization_cut_point(cut_point=1.05)

nf.current_parameters_chromatinization


# Chromatinization image parameters
nf.current_parameters_img_adj_chro

nf.set_adj_chrom_gamma(gamma=0.25)

nf.set_adj_chrom_contrast(contrast=4)

nf.set_adj_chrom_brightness(brightness=950)

nf.current_parameters_img_adj_chro


# Second execution of the third step (chromatinization analysis)
nf.nuclei_chromatinization()


###############################################################################

# Return results
chromatinization_results, analysed_chromatinization_img = (
    nf.get_results_nuclei_chromatinization()
)

###############################################################################


# If your parameters are correct for your data, you can run series analysis on more images


# Chromatinization CTRL CELLS
series_results_chromatinization = nf.series_analysis_chromatinization(
    path_to_images="test_data/flow_cytometry/ctrl",
    file_extension="tif",
    selected_id=[],
    selection_opt=True,
    include_img=False,
    test_series=0,
)

# Chromatinization DISEASE CELLS
series_results_chromatinization2 = nf.series_analysis_chromatinization(
    path_to_images="test_data/flow_cytometry/dis",
    file_extension="tif",
    selected_id=[],
    selection_opt=True,
    include_img=False,
    test_series=0,
)
###############################################################################

from jimg_ncd.nuclei import NucleiDataManagement

# initiate class with NucleiFinder data
ndm = NucleiDataManagement(series_results_chromatinization, "healthy_chromatinization")
ndm2 = NucleiDataManagement(
    series_results_chromatinization2, "disease_chromatinization"
)

# get data as data frame
df = ndm.get_data()
print(df)

ndm.save_results_df(path="")


df2 = ndm2.get_data()
print(df)

ndm2.save_results_df(path="")


# load IS data
import pandas as pd

healthy_data = pd.read_csv("test_data/flow_cytometry/ctrl.txt", sep="\t", header=1)

disease_data = pd.read_csv("test_data/flow_cytometry/dis.txt", sep="\t", header=1)

# select data with cell size info
selectes_columns = [
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

ndm.add_IS_data(healthy_data, IS_features=selectes_columns)


ndm2.add_IS_data(disease_data, IS_features=selectes_columns)


df_is = ndm.get_data_with_IS()
print(df_is)

ndm.save_results_df_with_IS(path="")


df_is2 = ndm2.get_data_with_IS()
print(df_is2)

ndm2.save_results_df_with_IS(path="")


# save projects
# NOTE:
# Image Stream (IS) data is NOT saved within the .nuc project file.
# After loading a project using `load_nuc_dict()`, IS data must be added again
# using the `add_IS_data()` method.

ndm.save_nuc_project(path="")

ndm2.save_nuc_project(path="")


###############################################################################

from jimg_ncd.nuclei import NucleiDataManagement

ndm = NucleiDataManagement.load_nuc_dict("healthy_chromatinization.nuc")
ndm2 = NucleiDataManagement.load_nuc_dict("disease_chromatinization.nuc")


# load IS data
import pandas as pd

healthy_data = pd.read_csv("test_data/flow_cytometry/ctrl.txt", sep="\t", header=1)

disease_data = pd.read_csv("test_data/flow_cytometry/dis.txt", sep="\t", header=1)

# select data with cell size info
selectes_columns = [
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

ndm.add_IS_data(healthy_data, IS_features=selectes_columns)


ndm2.add_IS_data(disease_data, IS_features=selectes_columns)


ndm.add_experiment([ndm2])

df = ndm.get_mutual_experiments_data(inc_is=True)
print(df)

ndm.save_mutual_experiments(path="", inc_is=True)

###############################################################################

import pandas as pd

from jimg_ncd.nuclei import GroupAnalysis

data = pd.read_csv(
    "healthy_chromatinization_disease_chromatinization_IS.csv", sep=",", header=0
)

# initiate class
ga = GroupAnalysis.load_data(data, ids_col="id_name", set_col="set")


# check available groups for selection of differential features
ga.groups

# run DFA analysis on example sets

ga.DFA(
    meta_group_by="sets",
    sets={
        "disease": ["disease_chromatinization"],
        "ctrl": ["healthy_chromatinization"],
    },
    n_proc=5,
)

group_diff_features = ga.get_DFA()

ga.heatmap_DFA()

fig = ga.get_DFA_plot()

fig.savefig("DFA_groups.png", dpi=300, bbox_inches="tight")

###############################################################################

# select differential features

# diff_features = list(group_diff_features['feature'][group_diff_features['p_val'] <= 0.05])

# ga.select_data(features_list = diff_features)

###############################################################################

# scale data
ga.data_scale()


# run PCA dimensionality reduction
ga.PCA()


# get PCA data, if required
pca_data = ga.get_PCA()


# run PC variance analysis
ga.var_plot()


# get var_data, if required
var_data = ga.get_var_data()


# get knee_plot, if required
knee_plot = ga.get_knee_plot(show=True)

knee_plot.savefig("knee_plot.png", dpi=300)

###############################################################################


# run UMAP dimensionality reduction
ga.UMAP(
    PC_num=8,
    factorize_with_metadata=False,
    harmonize_sets=True,
    n_neighbors=10,
    min_dist=0.01,
    n_components=2,
)


# get UMAP_data, if required
UMAP_data = ga.get_UMAP_data()


# get UMAP_plots, if required
UMAP_plots = ga.get_UMAP_plots(show=True)

UMAP_plots.keys()

UMAP_plots["PrimaryUMAP"].savefig("UMAP.png", dpi=300)
###############################################################################


# run db_scan on UMAP components
ga.db_scan(eps=0.5, min_samples=10)


# run UMAP_on_clusters
ga.UMAP_on_clusters(min_entities=5)


# get UMAP_plots, if required
UMAP_plots = ga.get_UMAP_plots(show=True)

UMAP_plots.keys()

UMAP_plots["ClusterUMAP"].savefig("UMAP_clusters.png", dpi=300)
UMAP_plots["ClusterXSetsUMAP"].savefig("ClusterXSetsUMAP.png", dpi=300)
###############################################################################


# get full_data [data + metadata], if required
full_data = ga.full_info()

###############################################################################

# check available groups for selection of differential features
ga.groups


# run DFA analysis on finl clusters
ga.DFA(meta_group_by="full_name", sets={}, n_proc=5)

dfa_clusters = ga.get_DFA()


ga.heatmap_DFA(top_n=3)

fig = ga.get_DFA_plot()


fig.savefig("DFA_clusters.png", dpi=300, bbox_inches="tight")

###############################################################################


ga.print_avaiable_features()


ga.proportion_analysis(
    grouping_col="sets", val_col="nuclei_per_img", grouping_dict=None, omit=None
)


pl = ga.get_proportion_plot(show=True)

pl.savefig("proportion.png", dpi=300, bbox_inches="tight")

ga.get_proportion_stats()
