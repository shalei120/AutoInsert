while getopts "m:n:" arg
do
	case $arg in
		m)  echo "m : $OPTARG"
		  method=$OPTARG
      ;;
    n)  echo "n : $OPTARG"
		  name=$OPTARG
      ;;
    ?)
			echo "unknown argument"
	esac
done
runai submit $name -i anibali/pytorch -g 1 --backoff-limit 0 -v /mnt/nfs-2/lei/AutoInsert:home/lei/AutoInsert --working-dir /home/lei/AutoInsert --command -- bash submitdgx/$method.sh



