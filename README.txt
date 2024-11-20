Use overleaf.com, easiest online LaTex compiler that I have found. Provides tutorials too.


FILE SYSTEM:
   Chapter files in order of apprearence in PDF (all are called from Main.tex):
	Title
	Acknowledgements
	Glossary
	Chapter1
	Chapter2
	Chapter3
	Chapter4
	Chapter5
	Chapter6
	Chapter7
	Chapter8
	Appendix
	References
   Runner Files:
	CRCE-BE-Report.cls # Class file for defining formatting of the report.
	Main.tex
	References.bib # Called in Main.tex
	scilab.sci # Called in Appendix.tex

--------------------------------------------------------------------------------------------
	
Instructions to use this LaTex template

A. Change the author names, roll numbers, department, etc in the Title.tex file.

0. Project title:
\title{Project Title With Every First Letter in Upper Case}

1. student names:
\stua{Kenneth H. Rosen}
\stub{S. Sudarshan}
\stuc{Vaibhav V. Godbole}

2. student roll numbers:
\rollnuma{1938}
\rollnumb{1965}
\rollnumc{1111}

3. department:
\department{Bachelor of Engineering}

4. guide:
\setguide{Donald E. Knuth}

5. date
\date{\large October 20, 2015} %\today


B. make the following changes in the Acknowledgement.tex file

1. Name of HOD and Department
replace #Prof. Mahesh Sharma, Head of Information Technology Dept.# with your values
\noindent We thank Prof. Mahesh Sharma, Head of Information Technology Dept., Principal and the management of C.R.C.E., Mumbai for encouragement and providing necessary infrastructure for pursuing the project.\\


C. Make the following changes in Main.tex
This is the list of files that contain each of your chapters. you can add more using /include{}. add or remove chapters as per your need. the index will automatically be updated.

\include{Introduction.tex}
\input{literature_review.tex}
\input{modifications.tex}
\input{test.tex}
\input{system.tex}
\input{implement.tex}
\input{summary.tex}
\input{intro.tex}

D. Make following changes in CRCE-BE-Report.cls
1. Change names of Guide, HOD and Department

  \begin{center}\begin{tabular}{p{4cm} p{5cm} p{5cm} }
   & Prof. Swapnali Makdey\newline Guide  &\\ 
   && \\
   && \\
   && \\
   && \\
   && \\
   Dr. Jagruti Save\newline Head of Department, \newline AI and Data Science & & Dr. S. S. Rathod\newline Principal \\
  \end{tabular}\end{center}

2. Change your department on line 235
   \begin{flushleft}
   This is to certify that the project entitled \textbf{"\@title"} is a bonafide work of  \textbf{\@stua (\@rollnuma), \@stub (\@rollnumb), \@stuc (\@rollnumc)} \, submitted to the University of Mumbai in partial fulfillment of the requirement for the award of the degree of \textbf{Bachelor of Engineering} in \textbf{Artificial Intelligence and Data Science}.
   \end{flushleft} 



#=