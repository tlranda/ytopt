ss=0
for i in {1..4}
do
	start_time=$[ $(date +%s000) ]
	$* >/dev/null 
	end_time=$[ $(date +%s000) ]
	tt=$[ $[$end_time - $start_time] / 1000 ]
	#echo "execution time was $tt s".
        ss=$[ $ss + $tt ]
done
ptime=`echo "scale=2;$ss/4" | bc`;
tmp=`echo "$ptime" | cut -d '.' -f 1`;
if [ -z "$tmp" ]; then
     ptime="0$ptime";
fi;
echo "$ptime";

