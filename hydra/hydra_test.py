import hydra


def some_print(x, driver, pw, user):
    print(x)
    print(driver)
    print(pw)
    print(user)


@hydra.main(config_path="config.yaml")
def my_app(cfg):
    print(cfg.db.pretty())
    driver = cfg.db.driver
    user = cfg.db.user

    some_print(x="h", **cfg.db)


if __name__ == "__main__":
    my_app()



