import autordf2gml

#to run topology-based transformation
autordf2gml.topology_feature("config-soa-tb.ini") 

#to run content-based transformation only using simple-edges
autordf2gml.simpleedges_feature("config-aifb-cb-simple.ini")

#to run content-based transformation
autordf2gml.content_feature("config-soa-cb.ini") 
