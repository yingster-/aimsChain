j = 63
if (!exist("i")) i = 0
set palette rgbformulae 33,13,10
set xrange[0.4:3.4]
set yrange[-2:2]

p 'pes.dat' index 0 with image title '', \
'pes.dat' index 2::1 w l lc 'black' lw 1.5 title '',\
'pes.dat' index 1 using 2:3 w l lw 1.5 title '', \
'string.dat' index i using 2:3 w linespoints lw 1.5 lt 3 title "STRING"


if (i==0) pause 1
if (i<j) i=i+1;print i;pause 0.1;reread
if (i>=j) pause -1



