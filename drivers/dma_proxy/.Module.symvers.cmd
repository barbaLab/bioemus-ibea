cmd_/home/ubuntu/bioemus/drivers/dma_proxy/Module.symvers := sed 's/\.ko$$/\.o/' /home/ubuntu/bioemus/drivers/dma_proxy/modules.order | scripts/mod/modpost -m -a  -o /home/ubuntu/bioemus/drivers/dma_proxy/Module.symvers -e -i Module.symvers   -T -
