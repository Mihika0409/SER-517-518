
use Trauma;
 
 
 create table Trauma_NewData ( tid Bigint(10) , hosp_date date, age int, gender char(1), 
                           levels varchar(3), icd_code varchar (200),icd_code_9 varchar (200),trauma_type varchar(10),physical_abuse varchar (5),
                           airbag_deploy varchar(15),patient_pos varchar(15),
                           safety_equip_issues varchar(30), child_restraint varchar(10),mv_speed  varchar(10), 
                           fall_height varchar(10), transport_type varchar(100),transport_mode  varchar(50),feild_SBP int ,
                           feild_HR int, feild_RR int, resp_assis varchar(50),RTS int ,
                           feild_GCS int,arrived_from varchar(20),ED_LOS int,disposition varchar(10),ED_SBP int, 
                           ED_HR int, ED_RR int, ED_GCS int, total_vent_days varchar(5), days_in_icu varchar(5),  
                           hosp_LOS varchar(5),total_LOS varchar(5),received_blood varchar (5), brain_injury varchar(5),
                           time_to_first_OR varchar(15) , death varchar (5),discharge_dispo varchar (30),AIS varchar(10),AIS_2005 bigint);
                           
                           
                           
 LOAD DATA LOCAL INFILE '/Users/gowtham/Downloads/sixyears_newdata.csv' 
 INTO TABLE  Trauma.Trauma_NewData FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
 
 
 SELECT * 
FROM Trauma.Trauma_NewData
where tid <>0 and icd_code like '% fall%' 
or icd_code_9 like '% fall%' 
and transport_mode <> 'Private/Public Vehicle/Walk-in'
limit  50000
