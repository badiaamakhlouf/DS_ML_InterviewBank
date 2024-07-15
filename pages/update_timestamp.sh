#!/bin/bash
TIMESTAMP=$(date +"Last updated: %Y-%m-%d %H:%M:%S")
sed -i "s/Last updated:.*/$TIMESTAMP/" feature_engineering.md
