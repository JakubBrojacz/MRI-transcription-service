# MRI-transcription-service

# Julius model from 
https://sourceforge.net/projects/juliusmodels/files/


Od (12)


# Now the trace_matrix. The edges of the backtrace are encoded
# binary: 
# 1 = open gap in seqA, 
# 2 = match/mismatch of seqA and seqB, 
# 4 = open gap in seqB, 
# 8 = extend gap in seqA, and
# 16 = extend gap in seqB. 
# This values can be summed up.
# Thus, the trace score 7 means that the best score can either
# come from opening a gap in seqA (=1), pairing two characters
# of seqA and seqB (+2=3) or opening a gap in seqB (+4=7).
# However, if we only want the score we don't care about the trace.