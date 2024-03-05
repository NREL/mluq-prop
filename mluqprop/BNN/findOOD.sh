#mpiexec -np 6 python find_nearest_B_Le.py --input inputs/bestLRM10D_OOD.in
mpiexec -np 2 python find_nearest_D_Le.py --input inputs/bestLRM10D_OOD.in
#python find_nearest_D_Le.py --input inputs/bestLRM10D_OOD.in
#python make_uq_contours_D_Le_OOD.py --input inputs/bestLRM10D_OOD.in
