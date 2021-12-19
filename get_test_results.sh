#!/bin/bash

declare -a ALGOS=("PureSVD" "ALS" "SLIMBPR" "ItemKNN" "P3Alpha" "CAAE" "CFGAN" "GANMF" "DisGANMF")
declare -a DATASETS=("1M" "LastFM" "hetrec2011")
declare -a SIMILARITIES=("cosine" "tversky" "asymmetric" "dice" "jaccard" "euclidean")

for algo in "${ALGOS[@]}"; do
  for dataset in "${DATASETS[@]}"; do

    if [[ "$algo" == "GANMF" ]] || [[ "$algo" == "CFGAN" ]] || [[ "$algo" == "DisGANMF" ]]; then
      echo "$algo $dataset --user"
      python RunBestParameters.py "$algo" "$dataset" --user

      echo "$algo $dataset --item"
      python RunBestParameters.py "$algo" "$dataset" --item

    elif [[ "$algo" == "ItemKNN" ]]; then

      for sim in "${SIMILARITIES[@]}"; do
        echo "$algo $dataset --$sim"
        python RunBestParameters.py "$algo" "$sim" "$dataset"
      done

    else
      echo "$algo $dataset"
      python RunBestParameters.py "$algo" "$dataset"
    fi

  done
done
