
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '/media/lidan/ssd2/CDData/LEVIR-CD256/'
        elif data_name == 'DSIFN':
            self.label_transform = "norm"
            self.root_dir = '/media/lidan/ssd2/CDData/DSIFN_256/'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = '/media/lidan/ssd2/CDData/WHU-CD-256/'
        elif data_name == 'CDD':
            self.label_transform = "norm"
            self.root_dir = '/media/lidan/ssd2/CDData/CDD-CD-256/'
        elif data_name == 'TYPO':
            self.label_transform = "norm"
            self.root_dir = '/media/lidan/ssd2/CDData/TYPO/'
        elif data_name == 'quick_start_LEVIR':
            self.root_dir = './samples_LEVIR/'
        elif data_name == 'quick_start_DSIFN':
            self.root_dir = './samples_DSIFN/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

