import torch
import pyro
from pyro.ops.indexing import Vindex
import pyro.distributions as dist
from pyro.distributions import transforms


class Model():

    def __init__(self, n_c, model_name = None):
        self.n_c = n_c
        self.model_name = model_name
        print(f"{self.model_name} n_c: {n_c}")

        if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def model(self, data=None):

        ## PRIOR
        cont_sites = [site for site in data["mask"].keys() if site in ["SC", "SW", "VA", "EMB"]]
        disc_sites = [site for site in data["mask"].keys() if site in ["GI", "HL", "MP"]]
        data_points = data["mask"][(cont_sites + disc_sites)[0]].shape[0]

        cont_site_params = {}
        disc_site_params = {}

        ## continous
        for cont_site in cont_sites:
            base_c = dist.Normal(torch.ones(self.n_c) * (-1.0), torch.ones(self.n_c) * 1.0)
            #base_c = dist.Normal(torch.ones(self.n_c) * (-5.0), torch.ones(self.n_c) * (+10.0))
            cont_site_params[cont_site  + "_mode_c"] = pyro.sample(cont_site  + "_mode_c", dist.TransformedDistribution(base_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))
            cont_site_params[cont_site + "_conc_c"] = pyro.sample(cont_site + "_conc_c", dist.Gamma(torch.ones(self.n_c), torch.ones(self.n_c)).to_event(1))
            #cont_site_params[cont_site + "_conc_c"] = pyro.sample(cont_site + "_conc_c", dist.Gamma(torch.ones(self.n_c), torch.ones(self.n_c) * 10).to_event(1))


        ## discrete
        for disc_site in disc_sites:
            base_c = dist.Normal(torch.ones(self.n_c) * (-1.0), torch.ones(self.n_c) * 1.0)
            #base_c = dist.Normal(torch.ones(self.n_c) * (-5.0), torch.ones(self.n_c) * (+10.0))
            disc_site_params[disc_site + "_p_c"] = pyro.sample(disc_site + "_p_c", dist.TransformedDistribution(base_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))

        pi_Z_c = pyro.sample("pi_Z_c", dist.Dirichlet(torch.ones(self.n_c) / self.n_c))

        ## LIKELIHOOD
        with pyro.plate('data_plate', data_points):

            Z = pyro.sample('Z', dist.Categorical(pi_Z_c), infer={"enumerate": "parallel"})

            ## continous
            for cont_site in cont_sites:
                mode_c = Vindex(cont_site_params[cont_site  + "_mode_c"])[..., Z.long()]
                conc_c = Vindex(cont_site_params[cont_site + "_conc_c"])[..., Z.long()] #+ 2.0
                # https://en.wikipedia.org/wiki/Beta_distribution see mode and concentration section
                pyro.sample(cont_site, dist.Beta(1 + (conc_c * mode_c), 1 + (conc_c * (1 - mode_c))).mask(data["mask"][cont_site]),obs=data["data"][cont_site])
                #pyro.sample(cont_site, dist.Beta(mode_c * (conc_c-2)+1, (1-mode_c)* (conc_c-2)+1).mask(data["mask"][cont_site]),obs=data["data"][cont_site])

            ## discrete
            for disc_site in disc_sites:
                pyro.sample(disc_site, dist.Binomial(probs=Vindex(disc_site_params[disc_site  + "_p_c"])[..., Z.long()], total_count=1).mask(data["mask"][disc_site]), obs=data["data"][disc_site])


if __name__ == "__main__":
    pass

