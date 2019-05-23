rename:
	rename 's/^/$(prefix)_/' dist_cos.csv dist_euc.csv
dot:
	for file in *.dot;do dot -Tpng -o $$file.png $$file;done
clean:
	rm -f *.png *.dot *.node2vec *.csv *walks.txt *.json *.h5f *.log
