# run all of the scripts that demonstrate BNN capabilities on toy data, first echoing the options to the terminal

# coarse graining uncertainty
python ./toys/collider.py --help
python ./toys/collider.py --input ./toys/collider.in --savedir "Figures/collider"

# extrapolatory uncertainty
python ./toys/extrap.py --help
python ./toys/extrap.py --input ./toys/extrap.in --savedir "Figures/extrap"

# espistemic vs aleatoric
python ./toys/uncertainty_forms.py --help
python ./toys/uncertainty_forms.py --input ./toys/uncertainty_forms.in --savedir "Figures/uncertainty"
