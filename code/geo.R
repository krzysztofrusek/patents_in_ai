require(readr)
require(tidyverse)
require(ggplot2)

read_csv("gen/geo.csv",col_names = c("iso_a2","frac")) ->geo

library(rnaturalearth)
library(rnaturalearthdata)

world <- ne_countries(scale = "medium", type= 'countries',returnclass = "sf")
Europe <- world[which(world$continent == "Europe"),]


eu.maps <- inner_join(Europe,geo, by="iso_a2") %>% mutate(label=round(frac,1))

ggplot(eu.maps) +
  geom_sf(aes(fill=frac))+
  geom_sf_text(aes(label=label),size=4)+
  coord_sf(xlim = c(-25,50), ylim = c(30,73), expand = FALSE) +
  theme_void()+
  scale_fill_viridis_c(trans = "log",breaks=c(1,10,100,1000))+
  theme(legend.text=element_text(size=12))

ggsave("gen/geo.svg")
ggsave("gen/geo.pdf")