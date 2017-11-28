
# creating a table for the 

create table Trauma_Firework
(
select * from Trauma.Trauma_processed
where icd_code like '%Discharge of firework%');

select 	tid,icd_code,injury_comments,levels, feild_SBP, feild_HR, 
			feild_RR, feild_GCS, ED_SBP,ED_HR,ED_RR,
			ED_GCS
from  Trauma.Trauma_Firework
where levels = 'N'