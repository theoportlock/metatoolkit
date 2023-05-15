#!/bin/bash
htlatex main.tex
make4ht -ue mybuild.mk4 *.tex
#soffice --convert-to docx *.html
