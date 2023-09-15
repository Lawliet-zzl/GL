for SEED in 1 2 3 4 5
do
	python main.py --seed ${SEED} --data "arrhythmia" --alg "CC" --alpha 0.5
done

