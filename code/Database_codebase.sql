# command for creating a database/ schema 
create schema Trauma;


use Trauma;

drop table Trauma_Data;

 
  # creating a table for importing the csv file in to the database table 
  
 create table Trauma_Data ( tid Bigint(10) , hosp_date date, age int, gender char(1), 
                           levels varchar(3), icd_code varchar (200),trauma_type varchar(10),physical_abuse varchar (5),
                           injury_comments varchar (1000),airbag_deploy varchar(15),patient_pos varchar(15),
                           safety_equip_issues varchar(30), child_restraint varchar(10),mv_speed  varchar(10), 
                           fall_height varchar(10), transport_type varchar(100),transport_mode  varchar(50),feild_SBP int ,
                           feild_HR int, feild_schok_ind int, feild_RR int, resp_assis varchar(50),RTS int ,
                           feild_GCS int,arrived_from varchar(20),ED_LOS int,disposition varchar(10),ED_SBP int, 
                           ED_HR int, ED_RR int, ED_GCS int, total_vent_days varchar(5), days_in_icu varchar(5),  
                           hosp_LOS varchar(5),total_LOS varchar(5),received_blood varchar (5), brain_injury varchar(5),
                           time_to_first_OR varchar(15) , death varchar (5),discharge_dispo varchar (30),AIS varchar(10),AIS_2005 bigint);

 
 # write alter table command for adding primarary index (primary key )
 
 # loading data files in to the databases 
 
 LOAD DATA LOCAL INFILE '/Users/gowtham/Downloads/newdata_trauma.csv' 
 INTO TABLE  Trauma.Trauma_Data FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
 
 

# query for exporting any table data to csv format 

SELECT *
INTO OUTFILE '/Users/gowtham/Downloads/ test.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
ESCAPED BY '\\'
LINES TERMINATED BY '\n'
FROM Trauma.Trauma_Data
where tid <>0;


# creating a table for storing the a partially cleaned data set for this phase 
create table  Trauma_processed 
(
SELECT * FROM Trauma.Trauma_Data
where tid <> 0 ) ;

# creating  tables for categorized data based on ICD 10 codes 
# 1... Assault 
create table  Trauma_Assault(

SELECT * FROM Trauma.Trauma_processed
where icd_code like '%ASSAULT%');

# 2... Fall

create table  Trauma_Fall(

SELECT * FROM Trauma.Trauma_processed
where icd_code like '%FALL%')
