# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:49:25 2026

@author: Stut0001

Version with 'cumulative' hotspots


"""


#%% Importing the data, preparing data and creating panel dataframe will be the same


import pandas as pd
from geopy.distance import geodesic
import numpy as np
from pyproj import Geod
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

#set wd
os.chdir(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP")


#%% 
### key variables

#checked for CBSA 500k versions
jp = pd.read_excel(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\BLOOM_DATA\BLOOM_DATA_2022_CLEAN\3DPjobs_cbsa_year.xlsx", index_col=0) #checked
new_jp = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\jp_3dp_cbsa_year_norp.csv") #checked
core_appl_pat = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\core_appl_patents_fractional_per_cbsa_year.csv") #checked #until beginning of 2024

#data that passed through QGIS (not checked yet)

patents_assignee_db = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\USPTO_PATENT_DATA\USPTO_PATENT_DATA_CLEAN\AMpatents_assignee_debeule_cbsa_year.csv", index_col=0) #checked #until beginning of 2024

CBSA_centroids = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Geodata\WIP\CBSA2015_centroids_500k_states.csv") #checked

totalpatents = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\USPTO_PATENT_DATA\USPTO_PATENT_DATA_CLEAN\patents_assignee_location_filtered_cbsa_2000_2024.csv", index_col=0) #checked #until beginning of 2024


### controls 

industry_cbsa = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\cbp_naics4_3dp_2002_2023_cbsa.csv") #checked

degrees_prof_grad = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\cbsa_educ_att_acs5_2009_2023.csv" ) #checked

estab_large = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\cbp_estab_large_cbsa_2006_2023.csv", dtype={'GEOID': str}) #checked

#universities = pd.read_excel(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\CBSA_rnd_1972_2023_shares.xlsx", index_col=0) #checked

unirnd_herd = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\cbsa2015_herd_unirnd_2010_2023.csv", index_col=0) #checked
unirnd_ard = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\cbsa2015_ard_unirnd_1972_2009.csv", index_col=0) #checked

tech_prox = pd.read_csv(r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\WIP\tech_prox_cbsa_year_2000_2024.csv") #checked

#%% Prepare data

#merge herd and ard unirnd data
unirnd = pd.concat([unirnd_herd, unirnd_ard], ignore_index=True)

#select those from original dataset

unique_db_patents = patents_assignee_db.drop_duplicates(subset='patent_id')

#rename
totalpatents = totalpatents.rename(columns={"total_patents":"total_cpc_patents"})
CBSA_centroids = CBSA_centroids.rename(columns={"GEOID_2":"STATE_ID"})
CBSA_centroids = CBSA_centroids.rename(columns={"NAME_2":"STATE_NAME"})
new_jp = new_jp.rename(columns={"year_posted":"year"})


#rename CBSACode column to CBSA_ID
jp = jp.rename(columns={"CBSACode":"CBSA_ID"})
new_jp = new_jp.rename(columns={"CBSAFP":"CBSA_ID"})
unique_db_patents = unique_db_patents.rename(columns={"CBSAFP":"CBSA_ID"})
CBSA_centroids = CBSA_centroids.rename(columns={"CBSAFP":"CBSA_ID"})
totalpatents = totalpatents.rename(columns={"CBSAFP":"CBSA_ID"})
#degrees_prof_grad = degrees_prof_grad.rename(columns={"CBSA Code":"CBSA_ID"})
unirnd = unirnd.rename(columns={"CBSAFP":"CBSA_ID"})
core_appl_pat = core_appl_pat.rename(columns={"CBSAFP":"CBSA_ID"})
industry_cbsa = industry_cbsa.rename(columns={"CBSA Code":"CBSA_ID"})

#CBSAs Puerto Rico, Alaska and Hawaii
CBSAs_PR = pd.DataFrame({"CBSA_ID": ['10380','27580','41900','32420','42180','38660','10260','17620','41980','11640','25020','17640']})
CBSAs_AK = pd.DataFrame({"CBSA_ID": ['11260','21820','27940','28540']})
CBSAs_HI = pd.DataFrame({"CBSA_ID": ['46520', '25900', '28180', '27980']})

#merge these CBSAs to be removed
CBSAs_remove = pd.concat([CBSAs_AK, CBSAs_HI, CBSAs_PR], ignore_index=True)

#change data type
CBSAs_remove['CBSA_ID'] = CBSAs_remove['CBSA_ID'].astype('Int64') 
unique_db_patents['CBSA_ID'] = unique_db_patents['CBSA_ID'].astype('Int64') 
estab_large['CBSA_ID'] = estab_large['CBSA_ID'].astype('Int64') 

# Then extract and filter
CBSA_centroids = CBSA_centroids[~CBSA_centroids['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]

#exclude these CBSAs from the original data
jp = jp[~jp['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
new_jp = new_jp[~new_jp['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]

unique_db_patents = unique_db_patents[~unique_db_patents['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
degrees_prof_grad = degrees_prof_grad[~degrees_prof_grad['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
unirnd = unirnd[~unirnd['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
totalpatents = totalpatents[~totalpatents['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
core_appl_pat = core_appl_pat[~core_appl_pat['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
industry_cbsa = industry_cbsa[~industry_cbsa['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
estab_large = estab_large[~estab_large['CBSA_ID'].isin(CBSAs_remove['CBSA_ID'])]
tech_prox = tech_prox[~tech_prox['CBSA_i'].isin(CBSAs_remove['CBSA_ID'])]
tech_prox = tech_prox[~tech_prox['CBSA_j'].isin(CBSAs_remove['CBSA_ID'])]


#change patent date into usable format
unique_db_patents['patent_date'] = pd.to_datetime(unique_db_patents['patent_date'])

# extract year from this format
unique_db_patents['year'] = unique_db_patents['patent_date'].dt.year


#%% Checks

techproxcheck = tech_prox.head(100000)


#%% Create panel dataframe

# count patents per CBSA × year
cbsa_year_patents = (
    unique_db_patents.groupby(['CBSA_ID','year','NAME'])
      .size()
      .reset_index(name='num_patents')
)

# all unique CBSAs from your reference dataframe
all_cbsas = CBSA_centroids[['CBSA_ID', 'NAME']].drop_duplicates()

# all unique years in your data, make sure it's a DataFrame
all_years = pd.DataFrame({'year': unique_db_patents['year'].unique()})

# Cartesian product of all CBSAs × all years
all_cbsa_years = (
    all_cbsas.assign(key=1)
    .merge(all_years.assign(key=1), on='key')
    .drop('key', axis=1)
)

# merge with your patent counts
cbsa_year_patents_full = (
    all_cbsa_years
    .merge(cbsa_year_patents, on=['CBSA_ID','NAME','year'], how='left')
    .fillna({'num_patents': 0})
    .astype({'num_patents':'int'})
    .sort_values(['NAME','year'])
    .reset_index(drop=True)
)

#further merge with ALL cpc patents per cbsa-year between 2000-2020
cbsa_year_patents_full = (
    cbsa_year_patents_full
    .merge(totalpatents, on=['CBSA_ID','year'], how='left')
    .fillna({'total_cpc_patents': 0})
    .astype({'total_cpc_patents':'int'})
    .sort_values(['NAME','year'])
    .reset_index(drop=True)
)

#further merge job postings, uni RnD and prof/grad degrees, industry, large establishment data per CBSA-year
cbsa_year_patents_full = (
    cbsa_year_patents_full
    .merge(jp, on=['CBSA_ID','year'], how='left')
    .merge(new_jp, on=['CBSA_ID','year'], how='left')
    .merge(unirnd, on=['CBSA_ID', 'NAME', 'year'], how='left')
    .merge(degrees_prof_grad, on=['CBSA_ID', 'year'], how='left')
    #.merge(industry_cbsa[['ESTAB_3DPNAICS_SHARE','EMP_3DPNAICS_SHARE',\
     #                     'ESTAB_SHARE_OTHER','EMP_SHARE_OTHER','ESTAB_SHARE_SERVICES','EMP_SHARE_SERVICES',\
      #                        'EMP_SHARE_RESEARCH_EDUCATION', 'ESTAB_SHARE_RESEARCH_EDUCATION','EMP_SHARE_MEDICAL',\
       #                       'ESTAB_SHARE_MEDICAL', 'EMP_SHARE_MANUFACTURING', 'ESTAB_SHARE_MANUFACTURING',\
        #                          'TOTAL_EMP','TOTAL_ESTAB','year','CBSA_ID']], on=['year','CBSA_ID'], how='left')
    .merge(industry_cbsa, on=['CBSA_ID', 'year'], how='left')
    .merge(estab_large, on=['CBSA_ID', 'year'], how='left')
)
           
#also merge the fractional patent counts
cbsa_year_patents_full = (
    cbsa_year_patents_full
    .merge(core_appl_pat, on=['CBSA_ID','NAME','year'], how='left')
    .fillna({'core_3dp_patents_fractional': 0})
    .fillna({'application_3dp_patents_fractional': 0})
)


#merge CBSA centroid coordinates with cbsa_year_patents_full
cbsa_year_patents_full = cbsa_year_patents_full.merge(CBSA_centroids[['CBSA_ID','INTPTLAT','INTPTLON','STATE_ID']], on='CBSA_ID', how='left')


#check EMP_SHARE_PROFESSIONAL_SERVICES
cbsa_check = cbsa_year_patents_full[['CBSA_ID', 'NAME','year', 'EMP_SHARE_SERVICES']]

#change inf values into nan
cbsa_year_patents_full = cbsa_year_patents_full.replace([np.inf], np.nan)


#%% Function while keeping hotspots 'alive'


#adapt function so that a technological proximity value is added in an additional column in the output dataframe. 
#This value has to be added based on the CBSA-hotspot relation which has to be the same as the CBSA_i and CBSA_j relation.
#So I only want the technological proximity values added to the resulting dataframe based on whether the CBSA-hotspot combination is the same as the CBSA_i and CBSA_j combination.
#these values can be extracted from a second dataframe which has to serve as input for the function



#also for 1- 2- and 3y lags
def aggregate_years_v5_cumhotspot(df, tech_prox_df, thresholds, thresholds_core, thresholds_appl):
    """
    df: dataframe with columns including ['CBSA_ID', 'NAME', 'year', ...]
    tech_prox_df: dataframe with columns ['year', 'CBSA_i', 'CBSA_j', 'tech_prox']
    thresholds / thresholds_core / thresholds_appl:
    """

    result = df.loc[df["year"].between(2005, 2025)].copy()
    #make sure the order of years is correct
    result = result.sort_values(["CBSA_ID", "year"])
    
    #rename column
    #result = result.rename(columns={"tech_hiring": "jp_3dp_cbsa_year"})
    result = result.rename(columns={"total_jp": "jp_3dp_cbsa_year"})
    
    #-----------------------------------------------------
    #calculate 3y moving averages
    
    #select columns to be used to calculate the MA
    MA_cols = {
    "num_patents": "3y_MA_num_3dp_patents",
    "application_3dp_patents_fractional": "3y_MA_application_3dp_patents_fractional",
    "core_3dp_patents_fractional": "3y_MA_core_3dp_patents_fractional",
    "total_cpc_patents": "3y_MA_total_cpc_patents_cbsa",
    }
    
    #3y MA for 3DP patents per CBSA
    for old_value, new_value in MA_cols.items():
        result[new_value] = (
            result
            .groupby("CBSA_ID")[old_value]
            .transform(lambda x: x.rolling(window=3, min_periods=3, center=False).mean())
            )
    
    #create total value columns 
    total_cols = {
    "num_patents": "patents_3dp_total_year",
    "total_cpc_patents": "cpc_patents_total_year",
    "core_3dp_patents_fractional": "core_3dp_pat_total_year",
    "application_3dp_patents_fractional": "appl_3dp_pat_total_year",
    }
    
    #first sum to have the total value for all CBSAs
    for old_value, new_value in total_cols.items():
        result[new_value] = result.groupby("year")[old_value].transform("sum")

    #create MA total columns
    ma_total_cols = {
        "patents_3dp_total_year": "3y_MA_patents_3dp_total",
        "cpc_patents_total_year": "3y_MA_cpc_patents_total",
        "core_3dp_pat_total_year": "3y_MA_core_3dp_pat_total",
        "appl_3dp_pat_total_year": "3y_MA_appl_3dp_pat_total",
    }

    for old_value, new_value in ma_total_cols.items():
        result[new_value] = (
            result
            .groupby("year")[old_value]   # rolling over time, not CBSA
            .transform(lambda x: x.rolling(window=3, min_periods=3, center=False).mean())
            ) 

    #-------------------------------------------------------------
    ###RTA using 3y moving average
    
    #calculate RTA per CBSA-period, give RTA=0 according to the selected absolute thresholds for the number of 3DP patents in that period in the CBSA
    result["RTA"] = np.where(
        result["3y_MA_num_3dp_patents"] < result["year"].map(thresholds).fillna(0), 0,
        ((result["3y_MA_num_3dp_patents"]/ result["3y_MA_patents_3dp_total"]) / (result["3y_MA_total_cpc_patents_cbsa"] / result["3y_MA_cpc_patents_total"]))
        )

    result["RTA_core"] = np.where(
        result["3y_MA_core_3dp_patents_fractional"] <
        result["year"].map(thresholds_core).fillna(0),
        0,
        (result["3y_MA_core_3dp_patents_fractional"] / result["3y_MA_core_3dp_pat_total"]) /
        (result["3y_MA_total_cpc_patents_cbsa"] / result["3y_MA_cpc_patents_total"])
    )

    result["RTA_appl"] = np.where(
        result["3y_MA_application_3dp_patents_fractional"] <
        result["year"].map(thresholds_appl).fillna(0),
        0,
        (result["3y_MA_application_3dp_patents_fractional"] / result["3y_MA_appl_3dp_pat_total"]) /
        (result["3y_MA_total_cpc_patents_cbsa"] / result["3y_MA_cpc_patents_total"])
    )

    # ---------------------------------------------------------
    ###HOTSPOTS (YEARLY)

    #define hotspot based on the RTA (give 0 value to RTA smaller than 1 and 1 to everything else)
    result["hotspot"] = np.where(result["RTA"].fillna(0) < 1, 0, 1)
    result["hotspot_core"] = np.where(result["RTA_core"].fillna(0) < 1, 0, 1)
    result["hotspot_appl"] = np.where(result["RTA_appl"].fillna(0) < 1, 0, 1)

    # ---------------------------------------------------------
    ###MAKE HOTSPOTS CUMULATIVE

    result = result.sort_values(["CBSA_ID", "year"])

    for h in ["hotspot", "hotspot_core", "hotspot_appl"]:
        result[h] = (
            result
            .groupby("CBSA_ID", sort=False)[h]
            .cummax()
            )

    # ---------------------------------------------------------
    #DISTANCE TO CUMULATIVE HOTSPOTS (1–3Y LAGS)


    geod = Geod(ellps="WGS84")

    year_order = sorted(result["year"].unique())
    hotspot_types = ["hotspot", "hotspot_core", "hotspot_appl"]
    lags = [1, 2, 3]

    patent_col_map = {
        "hotspot": "3y_MA_num_3dp_patents",
        "hotspot_core": "3y_MA_core_3dp_patents_fractional",
        "hotspot_appl": "3y_MA_application_3dp_patents_fractional"
    }

    for hotspot_type in hotspot_types:
        for lag in lags:

            dist_col = f"dist_to_prev_{lag}y_{hotspot_type}_m"
            id_col = f"closest_{lag}y_{hotspot_type}_CBSA_ID"
            name_col = f"closest_{lag}y_{hotspot_type}_NAME"
            patents_col = f"3y_MA_patents_3dp_prev_{lag}y_{hotspot_type}"

            result[[dist_col, id_col, name_col, patents_col]] = np.nan

            for year in year_order:

                cutoff_year = year - lag

                prev_hotspots = result[
                    (result["year"] == cutoff_year) &
                    (result[hotspot_type] == 1)
                ]

                if prev_hotspots.empty:
                    continue

                curr_mask = result["year"] == year
                curr_indices = result[curr_mask].index

                prev_hotspot_ids = set(prev_hotspots["CBSA_ID"])

                already_hotspot = (
                    (result.loc[curr_indices, hotspot_type] == 1) &
                    (result.loc[curr_indices, "CBSA_ID"].isin(prev_hotspot_ids))
                )

                result.loc[curr_indices[already_hotspot], dist_col] = 0

                non_hotspot_indices = curr_indices[~already_hotspot]

                for idx in non_hotspot_indices:
                    row = result.loc[idx]

                    lons1 = np.repeat(row["INTPTLON"], len(prev_hotspots))
                    lats1 = np.repeat(row["INTPTLAT"], len(prev_hotspots))
                    lons2 = prev_hotspots["INTPTLON"].values
                    lats2 = prev_hotspots["INTPTLAT"].values

                    _, _, d = geod.inv(lons1, lats1, lons2, lats2)
                    min_idx = d.argmin()

                    src_col = patent_col_map[hotspot_type]

                    result.at[idx, dist_col] = d[min_idx]
                    result.at[idx, id_col] = prev_hotspots.iloc[min_idx]["CBSA_ID"]
                    result.at[idx, name_col] = prev_hotspots.iloc[min_idx]["NAME"]
                    result.at[idx, patents_col] = prev_hotspots.iloc[min_idx][src_col]


    #-----------------------------------------------------------
    #ADD TECHNOLOGICAL PROXIMITY VALUE HERE

    for hotspot_type in hotspot_types:
        for lag in lags:

            id_col = f"closest_{lag}y_{hotspot_type}_CBSA_ID"
            prox_col = f"tech_prox_{lag}y_{hotspot_type}"

            # Build lookup: match (year, CBSA_ID, closest_hotspot_CBSA_ID)
            # to (year, region_i, region_j) in tech_prox_df
            # Merge twice to handle both directions (i→j and j→i)
            
            # Create a lookup df with lagged year (year - lag)
            lookup = result[['year', 'CBSA_ID', id_col]].copy()
            lookup['year'] = lookup['year'] - lag  # ← shift year back by lag

            # Forward direction: CBSA_ID = region_i, hotspot = region_j
            merge_fwd = lookup.merge(
                tech_prox_df.rename(columns={
                    'CBSA_i': 'CBSA_ID',
                    'CBSA_j': id_col,
                    'tech_prox': prox_col
                }),
                on=['year', 'CBSA_ID', id_col],
                how='left'
            )[[prox_col]]

            # Reverse direction: CBSA_ID = region_j, hotspot = region_i
            merge_rev = lookup.merge(
                tech_prox_df.rename(columns={
                    'CBSA_j': 'CBSA_ID',
                    'CBSA_i': id_col,
                    'tech_prox': prox_col
                }),
                on=['year', 'CBSA_ID', id_col],
                how='left'
            )[[prox_col]]

            # Combine: take forward value, fill missing with reverse
            result[prox_col] = merge_fwd[prox_col].fillna(
                merge_rev[prox_col]
            ).values

    # ---------------------------------------------------------
    #LAGGED CORE VARIABLES (1–3 YEARS)

    core_lag_vars = {
        "total_unirnd": "unirnd_cbsa",
        #"Pop25+_graduate_prof_degree": "Pop25+_graduate_prof_degree",
        "prof_grad_degree_25+_share": "prof_grad_degree_25+_share",
        "3y_MA_num_3dp_patents": "3y_MA_num_3dp_patents",
        "ESTAB_LARGE_SHARE": "ESTAB_LARGE_SHARE"
    }

    for col, base in core_lag_vars.items():
        for lag in [1, 2, 3]:
            result[f"{base}_prev_{lag}y"] = (
                result.groupby("CBSA_ID")[col].shift(lag)
            )

    # ---------------------------------------------------------
    #INDUSTRY EMPLOYMENT & ESTABLISHMENTS ----------------

    industry_combos = {
        "RESEARCH_EDUCATION": ["6113", "5417", "6111", "6112"],
        "MEDICAL": ["6212", "3254", "6221"],
        "MANUFACTURING": ["3364", "3341", "3345", "3311"],
        "PROFESSIONAL_SERVICES": ["5413", "5414", "5416", "5112"],
    }

    for combo, codes in industry_combos.items():
        emp_cols = [f"EMP_{c}" for c in codes if f"EMP_{c}" in result.columns]
        est_cols = [f"ESTAB_{c}" for c in codes if f"ESTAB_{c}" in result.columns]

        result[f"EMP_{combo}"] = result[emp_cols].sum(axis=1) if emp_cols else 0
        result[f"ESTAB_{combo}"] = result[est_cols].sum(axis=1) if est_cols else 0

        result[f"EMP_SHARE_{combo}"] = result[f"EMP_{combo}"] / result["TOTAL_EMP"]
        result[f"ESTAB_SHARE_{combo}"] = result[f"ESTAB_{combo}"] / result["TOTAL_ESTAB"]

    all_emp = [
        f"EMP_{c}" for codes in industry_combos.values()
        for c in codes if f"EMP_{c}" in result.columns
    ]
    all_est = [
        f"ESTAB_{c}" for codes in industry_combos.values()
        for c in codes if f"ESTAB_{c}" in result.columns
    ]

    result["EMP_3DPNAICS_TOTAL"] = result[all_emp].sum(axis=1)
    result["EMP_3DPNAICS_SHARE"] = result["EMP_3DPNAICS_TOTAL"] / result["TOTAL_EMP"]

    result["ESTAB_3DPNAICS_TOTAL"] = result[all_est].sum(axis=1)
    result["ESTAB_3DPNAICS_SHARE"] = result["ESTAB_3DPNAICS_TOTAL"] / result["TOTAL_ESTAB"]

    # ---------------------------------------------------------
    # ---- LAG INDUSTRY VARIABLES (1–3 YEARS) -----------------
    # ---------------------------------------------------------

    lag_vars = (
        [f"EMP_{k}" for k in industry_combos] +
        [f"ESTAB_{k}" for k in industry_combos] +
        [f"EMP_SHARE_{k}" for k in industry_combos] +
        [f"ESTAB_SHARE_{k}" for k in industry_combos] +
        [
            "EMP_3DPNAICS_TOTAL", "EMP_3DPNAICS_SHARE",
            "ESTAB_3DPNAICS_TOTAL", "ESTAB_3DPNAICS_SHARE"
        ]
    )

    for v in lag_vars:
        for lag in [1, 2, 3]:
            result[f"{v}_prev_{lag}y"] = (
                result.groupby("CBSA_ID")[v].shift(lag)
            )

    return result





#%% check distributions


#define yearly thresholds for hotspots

thresholds_75 = {
    2007: 1.666667,
    2008: 1.666667,
    2009: 2.083333,
    2010: 2.250000,
    2011: 2.583333,  
    2012: 2.916667,  
    2013: 2.666667,
    2014: 2.500000,
    2015: 2.666667,  
    2016: 2.666667, 
    2017: 2.833333,  
    2018: 3.000000,
    2019: 3.083333,  
    2020: 3.666667,
    2021: 4.166667,
    2022: 4.666667,
    2023: 5.000000,
    2024: 5.333333,
}

thresholds_85 = {
    2007: 3.500000,
    2008: 5.516667,
    2009: 6.166667,
    2010: 7.350000,
    2011: 4.850000,
    2012: 4.933333,
    2013: 5.000000,
    2014: 5.250000,
    2015: 5.383333,
    2016: 5.466667,
    2017: 7.333333,
    2018: 7.750000,
    2019: 5.983333,
    2020: 7.666667,
    2021: 8.033333,
    2022: 9.383333,
    2023: 9.266667,
    2024: 11.716667,
}

thresholds_90 = {
    2007: 6.000000,
    2008: 6.633333,
    2009: 8.666667,
    2010: 11.633333,
    2011: 9.433333,
    2012: 9.033333,
    2013: 11.066667,
    2014: 8.166667,
    2015: 8.100000,
    2016: 7.000000,
    2017: 10.866667,
    2018: 11.933333,
    2019: 17.033333,
    2020: 18.933333,
    2021: 18.000000,
    2022: 17.433333,
    2023: 16.133333,
    2024: 18.100000,
}

thresholds_95 = {
    2007: 6.916667,
    2008: 10.916667,
    2009: 11.166667,
    2010: 12.966667,
    2011: 12.666667,
    2012: 12.716667,
    2013: 14.733333,
    2014: 17.583333,
    2015: 17.683333,
    2016: 18.333333,
    2017: 21.366667,
    2018: 25.883333,
    2019: 36.950000,
    2020: 47.266667,
    2021: 49.466667,
    2022: 44.783333,
    2023: 46.200000,
    2024: 46.800000,
}

thresholds_core_75 = {
    2007: 1.500000,
    2008: 2.277778,
    2009: 1.722222,
    2010: 1.277778,
    2011: 0.993056,
    2012: 1.000000,
    2013: 0.902778,
    2014: 1.000000,
    2015: 1.621212,
    2016: 1.841667,
    2017: 1.895833,
    2018: 2.000000,
    2019: 2.125000,
    2020: 2.581624,
    2021: 3.570437,
    2022: 3.587515,
    2023: 4.517689,
    2024: 4.396717,
}

thresholds_core_85 = {
    2007: 2.888889,
    2008: 3.955556,
    2009: 3.179167,
    2010: 3.222222,
    2011: 2.797222,
    2012: 1.583333,
    2013: 1.509167,
    2014: 2.419444,
    2015: 3.066389,
    2016: 3.825000,
    2017: 5.552698,
    2018: 5.949259,
    2019: 4.539722,
    2020: 6.212302,
    2021: 6.458056,
    2022: 6.643333,
    2023: 7.000556,
    2024: 7.747937,
}

thresholds_core_90 = {
    2007: 4.055556,
    2008: 4.794444,
    2009: 5.930556,
    2010: 5.700000,
    2011: 6.938889,
    2012: 5.161111,
    2013: 3.411667,
    2014: 3.066667,
    2015: 4.937778,
    2016: 5.825397,
    2017: 9.177778,
    2018: 8.732222,
    2019: 13.587222,
    2020: 15.412607,
    2021: 16.838694,
    2022: 12.196334,
    2023: 13.672751,
    2024: 15.009711,
}

thresholds_core_95 = {
    2007: 6.305556,
    2008: 9.516667,
    2009: 8.020833,
    2010: 7.916667,
    2011: 7.533333,
    2012: 7.111111,
    2013: 6.288889,
    2014: 6.386111,
    2015: 7.369048,
    2016: 10.437222,
    2017: 14.652725,
    2018: 19.690873,
    2019: 31.250625,
    2020: 36.311258,
    2021: 38.077169,
    2022: 39.671308,
    2023: 38.400413,
    2024: 40.660085,
}

thresholds_appl_75 = {
    2007: 1.250000,
    2008: 1.500000,
    2009: 1.708333,
    2010: 1.763889,
    2011: 2.055556,
    2012: 2.311111,
    2013: 2.211111,
    2014: 1.916667,
    2015: 2.000000,
    2016: 1.833333,
    2017: 1.722222,
    2018: 1.722222,
    2019: 1.323214,
    2020: 1.563294,
    2021: 1.530159,
    2022: 1.878472,
    2023: 1.833333,
    2024: 2.164683,
}

thresholds_appl_85 = {
    2007: 2.061111,
    2008: 2.241667,
    2009: 2.966667,
    2010: 2.855556,
    2011: 3.000000,
    2012: 3.888889,
    2013: 4.555556,
    2014: 4.458333,
    2015: 3.533333,
    2016: 2.993056,
    2017: 3.438889,
    2018: 3.626667,
    2019: 4.273492,
    2020: 3.155238,
    2021: 3.508519,
    2022: 2.996310,
    2023: 3.153333,
    2024: 3.725873,
}

thresholds_appl_90 = {
    2007: 2.522222,
    2008: 4.072222,
    2009: 3.988889,
    2010: 6.355556,
    2011: 5.751111,
    2012: 6.333333,
    2013: 6.722222,
    2014: 6.666667,
    2015: 5.483541,
    2016: 3.839408,
    2017: 4.163400,
    2018: 4.311481,
    2019: 5.341004,
    2020: 5.142500,
    2021: 5.719916,
    2022: 6.157526,
    2023: 6.409439,
    2024: 7.639437,
}

thresholds_appl_95 = {
    2007: 3.808333,
    2008: 4.833333,
    2009: 5.991667,
    2010: 7.292778,
    2011: 8.107778,
    2012: 8.691667,
    2013: 9.555556,
    2014: 10.898611,
    2015: 10.521270,
    2016: 8.250000,
    2017: 5.911429,
    2018: 6.652646,
    2019: 8.542050,
    2020: 8.704910,
    2021: 10.915498,
    2022: 11.798495,
    2023: 11.854446,
    2024: 12.397014,
}

cbsa_year_patents_full_year_core_appl = aggregate_years_v5_cumhotspot(cbsa_year_patents_full, tech_prox, thresholds_90, thresholds_core_90, thresholds_appl_90)


#%% Checks

distr_pat = cbsa_year_patents_full.groupby("year")["3y_MA_num_3dp_patents"]

distr_core = cbsa_year_patents_full.groupby("year")["3y_MA_core_3dp_patents_fractional"]
distr_appl = cbsa_year_patents_full.groupby("year")["3y_MA_application_3dp_patents_fractional"]


distr_core = (
    cbsa_year_patents_full_year_core_appl
    .loc[cbsa_year_patents_full_year_core_appl["3y_MA_core_3dp_patents_fractional"] > 0]
    .groupby("year")["3y_MA_core_3dp_patents_fractional"]
)
distr_core.describe()
distr_core.quantile(0.95)

distr_appl = (
    cbsa_year_patents_full_year_core_appl
    .loc[cbsa_year_patents_full_year_core_appl['3y_MA_application_3dp_patents_fractional'] > 0]
    .groupby("year")['3y_MA_application_3dp_patents_fractional']
)
distr_appl.describe()
distr_appl.quantile(0.95)


distr_3dppat = (
    cbsa_year_patents_full_year_core_appl
    .loc[cbsa_year_patents_full_year_core_appl["3y_MA_num_3dp_patents"] > 0]
    .groupby("year")["3y_MA_num_3dp_patents"]
)
distr_3dppat.describe()
distr_3dppat.quantile(0.90)



#count number of hotspots per year
hotspots_per_year = (
    cbsa_year_patents_full_year_core_appl[cbsa_year_patents_full_year_core_appl["hotspot_appl"] == 1]
    .groupby("year")['CBSA_ID'] 
    .nunique()
    .reset_index(name="num_hotspot_cbsas")
)


cbsa_year_patents_full_year_core_appl_select = (cbsa_year_patents_full_year_core_appl[['CBSA_ID','NAME','year','hotspot','dist_to_prev_1y_hotspot_m',
                                                                                       'closest_1y_hotspot_NAME','RTA','3y_MA_patents_3dp_prev_1y_hotspot',
                                                                                       '3y_MA_num_3dp_patents','EMP_3DPNAICS_SHARE_prev_1y']])

cbsa_year_patents_full_year_core_appl_select = (cbsa_year_patents_full_year_core_appl[['CBSA_ID','NAME','year','ESTAB_LARGE_SHARE','ESTAB_LARGE_SHARE_prev_1y']])


#%% Clean dataframe

#remove rows from years before 2010 and after 2019
cbsa_year_patents_full_year_core_appl = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] >= 2010) & (cbsa_year_patents_full_year_core_appl["year"] <= 2025)]

#remove this specific CBSA with weird values
cbsa_year_patents_full_year_core_appl.drop(cbsa_year_patents_full_year_core_appl[cbsa_year_patents_full_year_core_appl["CBSA_ID"] == 31060].index, inplace=True)

#divide rnd by 1 000 000
#cbsa_year_patents_full_year_core_appl["unirnd_cbsa_year_billions"] = (cbsa_year_patents_full_year_core_appl["unirnd_cbsa_year"].fillna(0) / 1_000_000_000)
cbsa_year_patents_full_year_core_appl["unirnd_cbsa_prev_1y_billions"] = (cbsa_year_patents_full_year_core_appl["unirnd_cbsa_prev_1y"].fillna(0) / 1_000_000_000) 
cbsa_year_patents_full_year_core_appl["unirnd_cbsa_prev_2y_billions"] = (cbsa_year_patents_full_year_core_appl["unirnd_cbsa_prev_2y"].fillna(0) / 1_000_000_000) 
cbsa_year_patents_full_year_core_appl["unirnd_cbsa_prev_3y_billions"] = (cbsa_year_patents_full_year_core_appl["unirnd_cbsa_prev_3y"].fillna(0) / 1_000_000_000) 

#divide number of degrees by 1000
#cbsa_year_patents_full_year_core_appl["prof_grad_degree_year_100k"] = (cbsa_year_patents_full_year_core_appl["Pop25+_graduate_prof_degree"].fillna(0) / 100_000) 
#cbsa_year_patents_full_year_core_appl["prof_grad_degree_prev_1y_100k"] = (cbsa_year_patents_full_year_core_appl["Pop25+_graduate_prof_degree_prev_1y"].fillna(0) / 100_000) 
#cbsa_year_patents_full_year_core_appl["prof_grad_degree_prev_2y_100k"] = (cbsa_year_patents_full_year_core_appl["Pop25+_graduate_prof_degree_prev_2y"].fillna(0) / 100_000) 
#cbsa_year_patents_full_year_core_appl["prof_grad_degree_prev_3y_100k"] = (cbsa_year_patents_full_year_core_appl["Pop25+_graduate_prof_degree_prev_3y"].fillna(0) / 100_000) 


"prof_grad_degree_25+_share"



#divide distance to hotspot by 200km
cbsa_year_patents_full_year_core_appl["dist_to_prev_1y_hotspot_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_1y_hotspot_m"].fillna(0) / 100_000)
cbsa_year_patents_full_year_core_appl["dist_to_prev_1y_hotspot_core_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_1y_hotspot_core_m"].fillna(0) / 100_000)
cbsa_year_patents_full_year_core_appl["dist_to_prev_1y_hotspot_appl_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_1y_hotspot_appl_m"].fillna(0) / 100_000)

cbsa_year_patents_full_year_core_appl["dist_to_prev_2y_hotspot_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_2y_hotspot_m"].fillna(0) / 100_000)
cbsa_year_patents_full_year_core_appl["dist_to_prev_2y_hotspot_core_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_2y_hotspot_core_m"].fillna(0) / 100_000)
cbsa_year_patents_full_year_core_appl["dist_to_prev_2y_hotspot_appl_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_2y_hotspot_appl_m"].fillna(0) / 100_000)

cbsa_year_patents_full_year_core_appl["dist_to_prev_3y_hotspot_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_3y_hotspot_m"].fillna(0) / 100_000)
cbsa_year_patents_full_year_core_appl["dist_to_prev_3y_hotspot_core_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_3y_hotspot_core_m"].fillna(0) / 100_000)
cbsa_year_patents_full_year_core_appl["dist_to_prev_3y_hotspot_appl_100km"] = (cbsa_year_patents_full_year_core_appl["dist_to_prev_3y_hotspot_appl_m"].fillna(0) / 100_000)

#fill nans in patents_3dp_prev_hotspot with 0s (this is for CBSAs that are hotspots themselves)
cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_1y_hotspot_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_1y_hotspot"].fillna(0) / 10)
cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_1y_hotspot_core_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_1y_hotspot_core"].fillna(0) / 10)
cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_1y_hotspot_appl_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_1y_hotspot_appl"].fillna(0) / 10)

cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_2y_hotspot_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_2y_hotspot"].fillna(0) / 10)
cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_2y_hotspot_core_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_2y_hotspot_core"].fillna(0) / 10)
cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_2y_hotspot_appl_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_2y_hotspot_appl"].fillna(0) / 10)

cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_3y_hotspot_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_3y_hotspot"].fillna(0) / 10)
cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_3y_hotspot_core_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_3y_hotspot_core"].fillna(0) / 10)
cbsa_year_patents_full_year_core_appl["MA_3y_patents_3dp_prev_3y_hotspot_appl_10"] = (cbsa_year_patents_full_year_core_appl["3y_MA_patents_3dp_prev_3y_hotspot_appl"].fillna(0) / 10)



#%% Clean it a bit more

cbsa_year_patents_full_year_core_appl = cbsa_year_patents_full_year_core_appl.drop(["INTPTLAT","INTPTLON","cpc_patents_cbsa_year",\
                                                                    "unirnd_cbsa_year","Pop25+_graduate_prof_degree","patents_3dp_total_year",\
                                                                    "cpc_patents_total_year", "RTA", "closest_hotspot_CBSA_ID",\
                                                                    "closest_hotspot_NAME", "dist_to_prev_hotspot_km", "unirnd_cbsa_prev_year",\
                                                                        "dist_to_prev_hotspot_core_km", "dist_to_prev_hotspot_appl_km", "closest_hotspot_core_NAME",\
                                                                            "closest_hotspot_appl_NAME", "closest_hotspot_core_CBSA_ID", "closest_hotspot_appl_CBSA_ID",\
                                                                                "RTA_core", "RTA_appl", "ESTAB_LARGE", "ESTAB_TOTAL"], axis=1, errors='ignore')

cbsa_year_patents_full_year_core_appl = cbsa_year_patents_full_year_core_appl.drop(['EMP_RESEARCH_EDUCATION', 'ESTAB_RESEARCH_EDUCATION', 'EMP_MEDICAL',\
 'ESTAB_MEDICAL', 'EMP_MANUFACTURING', 'ESTAB_MANUFACTURING', 'EMP_PROFESSIONAL_SERVICES', 'ESTAB_PROFESSIONAL_SERVICES',\
 'EMP_OTHER', 'ESTAB_OTHER', 'EMP_SHARE_RESEARCH_EDUCATION', 'ESTAB_SHARE_RESEARCH_EDUCATION',\
 'EMP_SHARE_MEDICAL', 'ESTAB_SHARE_MEDICAL', 'EMP_SHARE_MANUFACTURING', 'ESTAB_SHARE_MANUFACTURING',\
 'EMP_SHARE_PROFESSIONAL_SERVICES', 'ESTAB_SHARE_PROFESSIONAL_SERVICES', 'EMP_SHARE_OTHER', 'ESTAB_SHARE_OTHER', 'EMP_3DPNAICS_TOTAL',\
 'EMP_3DPNAICS_SHARE', 'ESTAB_3DPNAICS_TOTAL', 'ESTAB_3DPNAICS_SHARE', 'EMP_RESEARCH_EDUCATION_prev_1y',\
 'EMP_MEDICAL_prev_1y', 'EMP_MANUFACTURING_prev_1y', 'EMP_PROFESSIONAL_SERVICES_prev_1y', 'EMP_OTHER_prev_1y',\
 'ESTAB_RESEARCH_EDUCATION_prev_1y', 'ESTAB_MEDICAL_prev_1y', 'ESTAB_MANUFACTURING_prev_1y',\
 'ESTAB_PROFESSIONAL_SERVICES_prev_1y', 'ESTAB_OTHER_prev_1y',  'EMP_RESEARCH_EDUCATION',\
 'ESTAB_RESEARCH_EDUCATION', 'EMP_MEDICAL', 'ESTAB_MEDICAL', 'EMP_MANUFACTURING',\
 'ESTAB_MANUFACTURING', 'EMP_PROFESSIONAL_SERVICES', 'ESTAB_PROFESSIONAL_SERVICES', 'EMP_OTHER', 'ESTAB_OTHER',\
 'EMP_SHARE_RESEARCH_EDUCATION', 'ESTAB_SHARE_RESEARCH_EDUCATION', 'EMP_SHARE_MEDICAL',\
 'ESTAB_SHARE_MEDICAL', 'EMP_SHARE_MANUFACTURING', 'ESTAB_SHARE_MANUFACTURING',\
 'EMP_SHARE_PROFESSIONAL_SERVICES', 'ESTAB_SHARE_PROFESSIONAL_SERVICES', 'EMP_SHARE_OTHER',\
 'ESTAB_SHARE_OTHER', 'EMP_3DPNAICS_TOTAL', 'EMP_3DPNAICS_SHARE',\
 'ESTAB_3DPNAICS_TOTAL', 'ESTAB_3DPNAICS_SHARE', 'EMP_RESEARCH_EDUCATION_prev_1y',\
 'EMP_MEDICAL_prev_1y', 'EMP_MANUFACTURING_prev_1y', 'EMP_PROFESSIONAL_SERVICES_prev_1y',\
 'EMP_OTHER_prev_1y', 'ESTAB_RESEARCH_EDUCATION_prev_1y', 'ESTAB_MEDICAL_prev_1y',\
 'ESTAB_MANUFACTURING_prev_1y', 'ESTAB_PROFESSIONAL_SERVICES_prev_1y', 'ESTAB_OTHER_prev_1y',  'EMP_3254',\
  'EMP_3311',  'EMP_3331',  'EMP_3333',  'EMP_3339',  'EMP_3341',  'EMP_3345',  'EMP_3359',  'EMP_3364',\
  'EMP_3391',  'EMP_4236',  'EMP_4411',  'EMP_5112',  'EMP_5191',  'EMP_5241',  'EMP_5413',  'EMP_5414',\
  'EMP_5415',  'EMP_5416',  'EMP_5417',  'EMP_5419',  'EMP_5614',  'EMP_6111',  'EMP_6112',  'EMP_6113',\
  'EMP_6212',  'EMP_6221',  'EMP_7211',  'EMP_7212',  'ESTAB_3254',  'ESTAB_3311',  'ESTAB_3331',\
  'ESTAB_3333',  'ESTAB_3339',  'ESTAB_3341',  'ESTAB_3345',  'ESTAB_3359',  'ESTAB_3364',  'ESTAB_3391',\
  'ESTAB_4236',  'ESTAB_4411',  'ESTAB_5112',  'ESTAB_5191',  'ESTAB_5241',  'ESTAB_5413',  'ESTAB_5414',\
  'ESTAB_5415',  'ESTAB_5416',  'ESTAB_5417',  'ESTAB_5419',  'ESTAB_5614',  'ESTAB_6111',  'ESTAB_6112',\
  'ESTAB_6113',  'ESTAB_6212',  'ESTAB_6221',  'ESTAB_7211',  'ESTAB_7212'], axis=1, errors='ignore')


    
    
#%% Checks
    
nandataset = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl['tech_prox_1y_hotspot'].isna())]


nandataset = nandataset[['CBSA_ID', 'NAME',  'year',  'closest_1y_hotspot_CBSA_ID','closest_1y_hotspot_NAME','tech_prox_1y_hotspot']]

tech_prox_check = tech_prox.tail(100000)

    
#%% Export dataset

output_folder = r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\MODEL_DATA"

output_path = os.path.join(output_folder, "model_data_year_core_appl_cumhotspots_90thperc_newjpdata_2010_2023_norp_noeduc.csv")

cbsa_year_patents_full_year_core_appl.to_csv(output_path, index=False)



#%% For maps, select yearly data

cbsa_year_patents_full_year_core_appl_2010 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2010)]
cbsa_year_patents_full_year_core_appl_2011 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2011)]
cbsa_year_patents_full_year_core_appl_2012 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2012)]
cbsa_year_patents_full_year_core_appl_2013 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2013)]
cbsa_year_patents_full_year_core_appl_2014 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2014)]
cbsa_year_patents_full_year_core_appl_2015 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2015)]
cbsa_year_patents_full_year_core_appl_2016 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2016)]
cbsa_year_patents_full_year_core_appl_2017 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2017)]
cbsa_year_patents_full_year_core_appl_2018 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2018)]
cbsa_year_patents_full_year_core_appl_2019 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2019)]
cbsa_year_patents_full_year_core_appl_2020 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2020)]
cbsa_year_patents_full_year_core_appl_2021 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2021)]
cbsa_year_patents_full_year_core_appl_2022 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2022)]
cbsa_year_patents_full_year_core_appl_2023 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2023)]
cbsa_year_patents_full_year_core_appl_2024 = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] == 2024)]


#get CBSAs with zero job postings

# Filter for rows with 0 job postings
zero_jp = cbsa_year_patents_full_year_core_appl[
    cbsa_year_patents_full_year_core_appl["CBSA_ID"].map(
        cbsa_year_patents_full_year_core_appl.groupby("CBSA_ID")["jp_3dp_cbsa_year"].max()
    ) == 0
]

zero_jp_unique = zero_jp.drop_duplicates(subset="CBSA_ID")


#sum dataset to get values of job postings, patents,... per CBSA across all years in the study period
cbsa_year_patents_full_year_core_appl_summed = cbsa_year_patents_full_year_core_appl[(cbsa_year_patents_full_year_core_appl["year"] >= 2010) & (cbsa_year_patents_full_year_core_appl["year"] <= 2024)]
cbsa_year_patents_full_year_core_appl_summed = cbsa_year_patents_full_year_core_appl_summed.groupby('CBSA_ID')[['jp_3dp_cbsa_year', 'num_patents']].sum().reset_index()





#%% export


output_folder2 = r"C:\Users\Stut0001\OneDrive - Universiteit Utrecht\Documents\Data\Numdata\MODEL_DATA"
output_path2 = os.path.join(output_folder2, "cbsa_year_patents_full_year_core_appl_2010_2023_90th.csv")
cbsa_year_patents_full_year_core_appl_2022.to_csv(output_path2, index=False)










#%% checks


#in the tech_prox dataframe, give me the value of tech_prox for the year 2009, CBSA_i 10100 and CBSA_j 16980

tech_prox.loc[
    (tech_prox['year'] == 2009) &
    (
        (
            (tech_prox['CBSA_i'] == 10100) &
            (tech_prox['CBSA_j'] == 16980)
        ) |
        (
            (tech_prox['CBSA_i'] == 16980) &
            (tech_prox['CBSA_j'] == 10100)
        )
    ),
    'tech_prox'
].iloc[0]


#give me the tech_prox value for the year 2010 for CBSA_i 12220 and CBSA_j 14460
tech_prox.loc[
    (tech_prox['year'] == 2009) &
    (
        (
            (tech_prox['CBSA_i'] == 11220) &
            (tech_prox['CBSA_j'] == 14460)
        ) |
        (
            (tech_prox['CBSA_i'] == 14460) &
            (tech_prox['CBSA_j'] == 11220)
        )
    ),
    'tech_prox'
].iloc[0]














