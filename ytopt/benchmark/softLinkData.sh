#!/bin/bash

link_suffix='-tl';
experiment_suffix='_exp';

echo "NARGS: $#";
# Check args
if [[ $# -lt 1 ]]; then
	echo "TOO FEW ARGUMENTS";
	echo "USAGE: $0 <path/to/data> [link_suffix='${link_suffix}'] [experiment_suffix='${experiment_suffix}']";
	exit;
fi
datapath=$1;
if [[ $# -ge 2 ]]; then
	link_suffix=$2;
fi
if [[ $# -ge 3 ]]; then
	experiment_suffix=$3;
fi
if [[ $# -ge 4 ]]; then
	echo "TOO MANY ARGUMENTS";
	echo "USAGE: $0 <path/to/data> [link_suffix='${link_suffix}'] [experiment_suffix='${experiment_suffix}']";
	exit;
fi
# Announce args
echo "DATAPATH: ${datapath}";
echo "LINK SUFFIX: ${link_suffix}";
echo -e "EXPERIMENT SUFFIX: ${experiment_suffix}\n";

# Search for local dirs
e_Dirs=();
for f in `ls -d */`; do
	if [[ "$f" == *${experiment_suffix} || "$f" == *${experiment_suffix}/ ]]; then
		e_Dirs+="$f ";
	else
		echo "'${f}' does not end with '${experiment_suffix}'";
	fi
done;
echo "Located potential experimental directories:";
for e in ${e_Dirs[@]}; do
	echo -e "\t${e}";
done;
echo;

# Search for linkable dirs
l_Dirs=();
for f in `ls -d ${datapath}/*/`; do
	if [[ "$f" == *${link_suffix} || "$f" == *${link_suffix}/ ]]; then
		l_Dirs+="$f ";
	else
		echo "'${f}' does not end with '${link_suffix}'";
	fi
done;
echo "Located potential link directories:";
for l in ${l_Dirs[@]}; do
	echo -e "\t${l}";
done;
echo;

# Attempt to find matches
for e_d in ${e_Dirs[@]}; do
	# Trim / if present
	if [[ "${e_d}" == */ ]]; then
		trim_len=$((${#e_d}-1));
		e_d=${e_d::${trim_len}};
	fi
	exper=${e_d};
	# Trim _ if present
	if [[ "${e_d}" == _* ]]; then
		e_d=${e_d:1};
	fi
	# Trim experiment ending
	#echo "${e_d} --> ${#e_d} | ${experiment_suffix} --> ${#experiment_suffix}";
	trim_len=$((${#e_d}-${#experiment_suffix}));
	e_d=${e_d::${trim_len}};
	#echo "${e_d} --> ${#e_d}";
	# Convert - to _
	e_d=`echo ${e_d} | tr '-' '_'`;
	#echo "${e_d} --> ${#e_d}";
	found=0;
	for candidate in ${l_Dirs[@]}; do
		# Trim / if present
		if [[ "${candidate}" == */ ]]; then
			trim_len=$((${#candidate}-1));
			candidate=${candidate::${trim_len}};
		fi
		# Trim datapath/ if present
		if [[ "${candidate}" == *//* ]]; then
			trim_len=$((${#datapath}+1));
			candidate="${datapath}${candidate:${trim_len}}";
		fi
		# For comparison, drop datapath
		comparison=${candidate:${#datapath}};
		# Trim link ending
		trim_len=$((${#comparison}-${#link_suffix}));
		comparison=${comparison::${trim_len}};
		# Trim / if present
		if [[ "${comparison}" == /* ]]; then
			comparison=${comparison:1};
		fi
		# Trim _ if present
		if [[ "${comparison}" == _* ]]; then
			comparison=${comparison:1};
		fi
		# Convert - to _
		comparison=`echo ${comparison} | tr '-' '_'`;
		#echo "COMPARE ${e_d} vs ${comparison}"
		if [[ "${e_d}" == "${comparison}" ]]; then
			found=1;
			link=${candidate};
			break;
		fi
	done;
	if [[ ${found} -eq 1 ]]; then
		if [[ -x ${exper}/data ]]; then
			existing_link=(`ls -l ${exper}/data`);
			existing_link=${existing_link[${#existing_link}]};
			# Trim / if present
			if [[ "${existing_link}" == */ ]]; then
				trim_len=$((${#existing_link}-1));
				existing_link=${existing_link::${trim_len}};
			fi
			if [[ "${existing_link}" == "${link}" ]]; then
				echo "Proposed link for ${exper} already exists";
			else
				echo "Alternative link existed for ${exper}: ${existing_link}";
				echo "Would instead propose: ${link}";
			fi
		else
			echo "Create NEW link for ${exper}: ${link}";
			ln -s ${link} ${exper}/data;
		fi
	else
		echo "No link made for: ${exper}";
	fi
done;

