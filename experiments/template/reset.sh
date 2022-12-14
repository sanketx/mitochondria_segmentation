RED="\033[0;31m"
NC="\033[0m" # No Color

printf "Cleans the working directory for the experiment\n"
printf "${RED}WARNING:${NC} This will erase all experiment artefacts\n"
read -p "Are you sure you want to continue? [y/n]: " response

if [ $response == "y" ]
then
	rm -rvf checkpoints
	rm -rvf cmd_params
	rm -rvf csv_logs
	rm -rvf default_params
	rm -rvf wandb
	echo "Done"
fi
