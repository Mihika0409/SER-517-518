

# categorizing data based on ICD code child physical abuse : cretaing a Table for Child abuse data 

create table Trauma_Ch_Abuse
(
select * from Trauma.Trauma_processed
where icd_code like '%CHILD PHYSICAL ABUSE%');

# analysis of data based on child abuse 

select * 
from Trauma.Trauma_processed
where icd_code like '%CHILD PHYSICAL ABUSE%'
			and brain_injury = 'Y'
            and levels in ('N' ) 
            and death = 'D' ;


select * 
from Trauma.Trauma_processed
where icd_code like '%CHILD PHYSICAL ABUSE%'
						and brain_injury = 'Y'
						and levels in ('1' ) 
						and death = 'D' ;

select * 
from Trauma.Trauma_processed
where icd_code like '%CHILD PHYSICAL ABUSE%'
			and brain_injury = 'Y'
            and levels in ('2' ) 
            and death = 'D';

select * 
from Trauma.Trauma_processed
where icd_code like '%CHILD PHYSICAL ABUSE%'
			and brain_injury = 'Y'
            and levels in ('3' ) 
            and death = 'D';
