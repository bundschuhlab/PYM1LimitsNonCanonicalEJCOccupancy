# Copyright (C) <2025>  <The Ohio State University>       

# This program is free software: you can redistribute it and/or modify                              
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or    
# (at your option) any later version.                                                                                       
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of           
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       
# GNU General Public License for more details.                                                                             

# You should have received a copy of the GNU General Public License 
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Load required library
library(biomaRt)

# List available BioMart databases from Ensembl archive (April 2020 version)
listMarts(host='apr2020.archive.ensembl.org')

# Connect to the Ensembl database for human genes
ensembl100 = useMart(host='apr2020.archive.ensembl.org', biomart='ENSEMBL_MART_ENSEMBL', dataset='hsapiens_gene_ensembl')

# Retrieve available attributes and filters for this dataset
attributes <- listAttributes(ensembl100)
filters <- listFilters(ensembl100)

# Fetch MANE-Select transcript IDs
transcriptMANE <- getBM(attributes = c("transcript_mane_select","ensembl_transcript_id"),
                        mart = ensembl100,
                        useCache = FALSE)

# Filter out transcripts that are not MANE-Select
transcriptMANE <- subset(transcriptMANE, transcriptMANE$transcript_mane_select != "")

# Retrieve coding sequences for MANE-Select transcripts
sequences <- getBM(attributes = c("ensembl_transcript_id", "coding"),
                   values = transcriptMANE$ensembl_transcript_id,
                   filters = "ensembl_transcript_id",
                   mart = ensembl100,
                   useCache = FALSE)

# Retrieve 5' UTR sequences
UTR5sequences <- getBM(attributes = c("ensembl_transcript_id", "5utr"),
                       values = transcriptMANE$ensembl_transcript_id,
                       filters = "ensembl_transcript_id",
                       mart = ensembl100,
                       useCache = FALSE)

# Retrieve 3' UTR sequences
UTR3sequences <- getBM(attributes = c("ensembl_transcript_id", "3utr"),
                       values = transcriptMANE$ensembl_transcript_id,
                       filters = "ensembl_transcript_id",
                       mart = ensembl100,
                       useCache = FALSE)

# Load tidyverse for data manipulation
library(tidyverse)

# Extract the last three nucleotides of the coding sequence
sequences <- sequences %>% mutate(stop = str_extract(coding, "[:alpha:]{3}$"))

# Count occurrences of each stop codon
table(sequences$stop)

# Calculate lengths of 5' and 3' UTRs
UTR5sequences <- UTR5sequences %>% mutate(Length5UTR = str_length(`5utr`))
UTR3sequences <- UTR3sequences %>% mutate(Length3UTR = str_length(`3utr`))

# Retrieve exon-level details
Exons <- getBM(attributes = c("ensembl_gene_id", "ensembl_transcript_id", "ensembl_exon_id", "rank",
                              "5_utr_end", "3_utr_start", "cds_start", "cds_end", "cds_length",
                              "cdna_coding_start", "cdna_coding_end", "exon_chrom_start", "exon_chrom_end",
                              "phase", "end_phase", "strand"),
               filters = "ensembl_transcript_id",
               values = transcriptMANE$ensembl_transcript_id,
               mart = ensembl100,
               useCache = FALSE)

# Arrange exons by gene ID and rank
Exons <- Exons %>% arrange(ensembl_gene_id, rank)

# Count exons ending in different reading frames
exon_counts <- Exons %>%
  group_by(ensembl_gene_id, end_phase) %>%
  dplyr::summarize(exon_count = n()) %>%
  filter(end_phase != -1) %>%
  pivot_wider(names_from = end_phase, values_from = exon_count, values_fill = 0)
names(exon_counts) <- c("ensembl_gene_id", "F0", "F1", "F2")

# Calculate distance from start codon (AUG) to exon junctions
cdsexon1 <- Exons %>% filter(phase == -1 & end_phase != -1) %>%
  mutate(exonlength = abs(exon_chrom_end - exon_chrom_start),
         AUGtoExonJunction = cds_end,
         UTR5junctions = rank - 1)

# Distance from upstream exon junction to AUG
upcdsexon1 <- cdsexon1 %>% filter(rank > 1) %>% mutate(ExonJunctiontoAUG = exonlength - cds_end)

# Distance from exon junction to stop codon
downcdsexon1 <- Exons %>% filter(phase != -1 & end_phase == -1) %>%
  mutate(ExonJunctiontoStop = abs(cds_end - cds_start))

# Count 3' UTR junctions
UTR3exons <- Exons %>% filter(end_phase == -1 & !is.na(`3_utr_start`)) %>% filter(rank != 1)
UTR3exonsCount <- UTR3exons %>% group_by(ensembl_transcript_id) %>% dplyr::summarize(exon_count = n()) %>%
  mutate(UTR3junctions = exon_count - 1)

# Calculate average CDS exon length
cdsexon <- Exons %>% filter(is.na(`5_utr_end`) & is.na(`3_utr_start`)) %>%
  mutate(exonlength = abs(exon_chrom_start - exon_chrom_end))
average_exon_lengths <- cdsexon %>% group_by(ensembl_gene_id) %>%
  summarize(average_cds_exon_length = mean(exonlength, na.rm = TRUE))


# Retrieve gene IDs and names
genenames <- getBM(attributes = c("ensembl_gene_id", "external_gene_name", "ensembl_transcript_id", "tmhmm", "signalp", "percentage_gene_gc_content"),
                   mart = ensembl100, useCache = FALSE) %>%
  filter(ensembl_transcript_id %in% transcriptMANE$ensembl_transcript_id)
names(genenames)[2] <- "Gene_name"

#Joining all the information to get a Master table

Finaltable<-genenames[,c(1,3)]
genenames<-genenames%>%mutate(TMDomain=case_when(tmhmm == "TMhelix" ~ 1,
                                                 TRUE ~ 0))

genenames<-genenames%>%mutate(SignalPeptide=case_when(signalp == "SignalP-noTM" ~ 1,
                                                      signalp == "SignalP-TM" ~ 1,
                                                 TRUE ~ 0))


Finaltable<-genenames[,c(1,3,6:8)]

Finaltable<-left_join(Finaltable, sequences[,c(1,3)])
Finaltable$Value <- 1
Finaltable<-Finaltable%>%pivot_wider(names_from = stop,
                               values_from = Value)
Finaltable[is.na(Finaltable)] <- 0

Finaltable<-left_join(Finaltable, UTR3sequences[,c(1,3)])
Finaltable<-left_join(Finaltable, UTR5sequences[,c(1,3)])

Finaltable<-left_join(Finaltable,exon_counts)

Finaltable<-left_join(Finaltable,cdsexon1[,c(2,18,19)])
Finaltable<-left_join(Finaltable,upcdsexon1[,c(2,19)])

Finaltable<-left_join(Finaltable,downcdsexon1[,c(2,17)])
Finaltable<-left_join(Finaltable,UTR3exonsCount[,c(1,3)])
Finaltable<-left_join(Finaltable,average_exon_lengths)

Finaltable$UTR5junctions[is.na(Finaltable$UTR5junctions)] <- 0
Finaltable$exonlength<-NULL
Finaltable$UTR3junctions[is.na(Finaltable$UTR3junctions)] <- 0

Finaltable<-distinct(Finaltable)

write_tsv(Finaltable,"FeaturesTableV2.tsv",col_names = T)

#Finaltable$ensembl_gene_id[duplicated(Finaltable$ensembl_gene_id)]
#Finaltable <- Finaltable[!duplicated(Finaltable$ensembl_gene_id), ]


