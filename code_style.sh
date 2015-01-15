
for i in $@
do
    astyle  --indent=spaces=4  --convert-tabs  --pad-oper --suffix=none $i
done
