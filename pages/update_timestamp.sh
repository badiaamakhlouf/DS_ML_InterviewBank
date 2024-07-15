#!/bin/bash
DATE_ONLY=$(date +"Last updated: %Y-%m-%d")
sed -i "s/Last updated:.*/$DATE_ONLY/" feature_engineering.md
