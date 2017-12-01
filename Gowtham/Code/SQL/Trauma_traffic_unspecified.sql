
# categorizing the data based on ICD code driver and passenger unspecified traffic accidents;creating a table for unspecified traffic accidents 

create table Trauma_Unspec_Trf_Acc (
SELECT * FROM Trauma.Trauma_processed
where icd_code like '%CAR OCCUPANT (DRIVER) (PASSENGER) INJURED IN UNSPECIFIED TRAFFIC ACCIDENT, INITIAL ENCOUNTER%')

# Analysis based on traffic unspecified traffic accidents

select tid,icd_code,injury_comments,feild_SBP, 
			 feild_HR,feild_GCS,brain_injury,
			 time_to_first_OR ,ED_GCS,ED_SBP
from Trauma_Unspec_Trf_Acc
where levels = 'N'

select 	tid,icd_code,injury_comments,feild_SBP, 
			 feild_HR,feild_GCS,brain_injury,
			 time_to_first_OR ,ED_GCS,ED_SBP
from Trauma_Unspec_Trf_Acc
where levels =  1

select 	tid,icd_code,injury_comments,feild_SBP, 
			 feild_HR,feild_GCS,brain_injury,
			 time_to_first_OR ,ED_GCS,ED_SBP
from Trauma_Unspec_Trf_Acc
where levels = 2
