Repeat forever:
    for config in hc.top_configs(hc.get('population_size'))
        gs,ds = config.generators, config.discriminators
        crossover(gs)
        crossover(ds)
        mutate(gs)
        mutate(ds)
        graph = create_graph(gs, ds)
        for batch:
            g_loss,d_loss,d_loss_fake = run_graph(graph)
            hc.gan.cost(g_loss, d_loss, d_loss_fake)
        #persist results, compare with total population
        save_results
