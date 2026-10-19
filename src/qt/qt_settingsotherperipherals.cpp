/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Other peripherals configuration UI module.
 *
 * Authors: Joakim L. Gilje <jgilje@jgilje.net>
 *          Jasmine Iwanek <jriwanek@gmail.com>
 *
 *          Copyright 2021 Joakim L. Gilje
 *          Copyright 2025 Jasmine Iwanek
 */
#include <cstdint>
#include <cstdio>
#include <vector>
#ifdef USE_VFIO
#    include <algorithm>
#    include <cstring>

#    include <QDir>
#    include <QFile>
#    include <QHash>
#    include <QProcess>
#    include <QStandardItemModel>
#    include <QStringBuilder>
#endif

extern "C" {
#include <86box/86box.h>
#include <86box/device.h>
#include <86box/machine.h>
#include <86box/mcamem.h>
#include <86box/isamem.h>
#include <86box/isarom.h>
#include <86box/isartc.h>
#include <86box/unittester.h>
#include <86box/softpower.h>
#include <86box/novell_cardkey.h>
#ifdef USE_VFIO
#    include <86box/vfio.h>
#endif
}

/* Forward-declare globals that some C++ files expect directly. This fixes
   cases where header visibility/order in generated moc/uic files makes the
   symbol not visible; the declaration is a no-op at runtime. */
extern "C" { extern int softpower_enabled; }

#include "qt_settings_completer.hpp"

#include "qt_settingsotherperipherals.hpp"
#include "ui_qt_settingsotherperipherals.h"

#include "qt_deviceconfig.hpp"
#include "qt_models_common.hpp"

#include "qt_defs.hpp"
#ifdef USE_VFIO
#    include "qt_util.hpp"
#endif

static bool
hasIsaOrSidecarBus(int machineId)
{
    return machine_has_bus(machineId, MACHINE_BUS_ISA | MACHINE_BUS_SIDECAR) > 0;
}

#ifdef USE_VFIO
enum {
    ColumnCheck = 0,
    ColumnGroup,
    ColumnLocation,
    ColumnID,
    ColumnName
};

/* Kept on the checkbox column item. */
enum {
    GroupRole = Qt::UserRole, /* IOMMU group, -1 for a device that was not found */
    ROMEnabledRole,
    ROMFileRole
};

/* The IOMMU group decides what can be passed through: the host has to give
   up every device in a group to pass any of them through. Once a device in a
   group is checked, the others in it show partially checked (claimed, but
   not passed through), and cycle between that and checked until no device
   in the group is checked any more. */
class VFIODeviceModel : public QStandardItemModel {
public:
    using QStandardItemModel::QStandardItemModel;

    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;
};

bool
VFIODeviceModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if ((role != Qt::CheckStateRole) || (index.column() != ColumnCheck))
        return QStandardItemModel::setData(index, value, role);

    /* A view asks for checked from partially checked, and unchecked from checked. */
    auto      state = static_cast<Qt::CheckState>(value.toInt());
    const int group = index.data(GroupRole).toInt();

    /* A device that was not found is in no group. */
    if (group < 0)
        return QStandardItemModel::setData(index, static_cast<int>((state == Qt::Unchecked) ? Qt::Unchecked : Qt::Checked), role);

    QList<QModelIndex> siblings;
    bool               otherChecked = false;
    for (int row = 0; row < rowCount(); row++) {
        const QModelIndex sibling = this->index(row, ColumnCheck);
        if ((row == index.row()) || (sibling.data(GroupRole).toInt() != group))
            continue;
        siblings.append(sibling);
        if (sibling.data(Qt::CheckStateRole).toInt() == Qt::Checked)
            otherChecked = true;
    }

    if (otherChecked) {
        if (state == Qt::Unchecked)
            state = Qt::PartiallyChecked;
    } else if (state == Qt::PartiallyChecked) {
        state = Qt::Unchecked;
    }

    if (!QStandardItemModel::setData(index, static_cast<int>(state), role))
        return false;

    /* Claim the group on the first device checked, release it on the last unchecked. */
    for (const auto &sibling : siblings) {
        const int siblingState = sibling.data(Qt::CheckStateRole).toInt();
        if ((state == Qt::Checked) && (siblingState == Qt::Unchecked))
            QStandardItemModel::setData(sibling, static_cast<int>(Qt::PartiallyChecked), role);
        else if ((state == Qt::Unchecked) && (siblingState == Qt::PartiallyChecked))
            QStandardItemModel::setData(sibling, static_cast<int>(Qt::Unchecked), role);
    }

    return true;
}

static const QString iommuGroupsPath = QStringLiteral("/sys/kernel/iommu_groups");

/* A configured location as dddd:bb:dd.f, or as written if it does not parse. */
static QString
normalizedLocation(const char *location)
{
    char buf[16];
    if (vfio_parse_location(location, buf, sizeof(buf)))
        return QString(buf);
    return QString(location);
}

/* A vendor or device ID from sysfs ("0x8086") as 4 hex digits. */
static QString
readSysfsId(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        return QStringLiteral("????");

    QString id = QString::fromLatin1(file.readAll()).trimmed();
    if (id.startsWith(QStringLiteral("0x")))
        id = id.mid(2);
    return id.isEmpty() ? QStringLiteral("????") : id.rightJustified(4, '0');
}

/* Device names as lspci gives them, by dddd:bb:dd.f location. */
static QHash<QString, QString>
lspciNames()
{
    QHash<QString, QString> names;
    QProcess                lspci;

    lspci.start(QStringLiteral("lspci"), { QStringLiteral("-D") });
    if (!lspci.waitForFinished(5000)) {
        lspci.kill();
        lspci.waitForFinished();
        return names;
    }

    const auto lines = QString::fromLocal8Bit(lspci.readAllStandardOutput()).split('\n');
    for (const auto &line : lines) {
        const int space = line.indexOf(' ');
        if (space > 0)
            names.insert(line.left(space), line.mid(space + 1).trimmed());
    }

    return names;
}

static QModelIndex
appendDevice(QStandardItemModel *model, int group, const QString &location, const QString &id, const QString &name)
{
    auto *check = new QStandardItem;
    check->setCheckable(true);
    check->setEditable(false);
    check->setData(group, GroupRole);
    check->setData(false, ROMEnabledRole);
    check->setData(QString(), ROMFileRole);

    QList<QStandardItem *> row = { check,
                                   new QStandardItem((group >= 0) ? QString::number(group) : QStringLiteral("??")),
                                   new QStandardItem(location),
                                   new QStandardItem(QStringLiteral("[%1]").arg(id)),
                                   new QStandardItem(name) };
    for (auto *item : row)
        item->setEditable(false);
    model->appendRow(row);

    return check->index();
}
#endif

SettingsOtherPeripherals::SettingsOtherPeripherals(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::SettingsOtherPeripherals)
{
    ui->setupUi(this);

    /* Memory expansion cards: one shared set of four slots, fed from the
       ISA board list (isamem) on ISA machines and from the MCA board list
       (mcamem) on MCA machines. */
    for (uint8_t i = 0; i < ISAMEM_MAX; ++i) {
        scMemExpCard[i] = new SettingsCompleter(findChild<QComboBox *>(QString("comboBoxMemExpCard%1").arg(i + 1)), nullptr);
        memexp_cfg_changed[i] = 0;
    }

    for (uint8_t i = 0; i < ISAROM_MAX; ++i) {
        scIsaRomCard[i] = new SettingsCompleter(findChild<QComboBox *>(QString("comboBoxIsaRomCard%1").arg(i + 1)), nullptr);
        isarom_cfg_changed[i] = 0;
    }

    scRTC           = new SettingsCompleter(ui->comboBoxRTC, nullptr);
    isartc_cfg_changed         = 0;

    unittester_cfg_changed     = 0;
    softpower_cfg_changed      = 0;
    softpower_card_enabled     = softpower_enabled > 0;
    novell_keycard_cfg_changed = 0;

#ifdef USE_VFIO
    vfioModel = new VFIODeviceModel(this);
    vfioModel->setHorizontalHeaderLabels({ QString(), tr("Group"), tr("Location"), tr("ID"), tr("Name") });
    ui->treeViewVFIO->setModel(vfioModel);

    /* FileField fixes its own width; let this one stretch across the panel. */
    ui->fileFieldVFIOROM->setMinimumWidth(0);
    ui->fileFieldVFIOROM->setMaximumWidth(QWIDGETSIZE_MAX);
    ui->fileFieldVFIOROM->setFilter(tr("ROM images") % util::DlgFilter({ "bin", "rom", "vbi" }) % tr("All files") % util::DlgFilter({ "*" }, true));

    vfioEnumerate();
    vfioRestore();

    for (int i = 0; i < vfioModel->columnCount(); i++)
        ui->treeViewVFIO->resizeColumnToContents(i);

    connect(ui->treeViewVFIO->selectionModel(), &QItemSelectionModel::currentRowChanged, this,
            [this](const QModelIndex &, const QModelIndex &) { vfioUpdateControls(); });
    connect(vfioModel, &QAbstractItemModel::dataChanged, this,
            [this](const QModelIndex &, const QModelIndex &, const QVector<int> &roles) {
                if (roles.isEmpty() || roles.contains(Qt::CheckStateRole))
                    vfioUpdateControls();
            });

    /* The custom ROM applies to the selected device. */
    connect(ui->checkBoxVFIOCustomROM, &QCheckBox::toggled, this, [this](bool checked) {
        if (vfioUpdating)
            return;
        const QModelIndex index = ui->treeViewVFIO->currentIndex().siblingAtColumn(ColumnCheck);
        if (index.isValid())
            vfioModel->setData(index, checked, ROMEnabledRole);
        ui->fileFieldVFIOROM->setEnabled(checked && ui->checkBoxVFIOCustomROM->isEnabled());
    });
    auto romFileChanged = [this](const QString &fileName) {
        if (vfioUpdating)
            return;
        const QModelIndex index = ui->treeViewVFIO->currentIndex().siblingAtColumn(ColumnCheck);
        if (index.isValid())
            vfioModel->setData(index, fileName, ROMFileRole);
    };
    connect(ui->fileFieldVFIOROM, &FileField::fileSelected, this, romFileChanged);
    connect(ui->fileFieldVFIOROM, &FileField::fileTextEntered, this, romFileChanged);

    if (vfioModel->rowCount() > 0)
        ui->treeViewVFIO->setCurrentIndex(vfioModel->index(0, 0));
    vfioUpdateControls();
#else
    /* VFIO is only available on Linux, with the development branch. */
    ui->tabWidgetSound->removeTab(ui->tabWidgetSound->indexOf(ui->tabVFIO));
    delete ui->tabVFIO;
#endif

    onCurrentMachineChanged(machine);
}

SettingsOtherPeripherals::~SettingsOtherPeripherals()
{
    for (uint8_t i = 0; i < ISAMEM_MAX; ++i)
        delete scMemExpCard[i];

    for (uint8_t i = 0; i < ISAROM_MAX; ++i)
        delete scIsaRomCard[i];

    delete scRTC;

    delete ui;
}

void
SettingsOtherPeripherals::onCurrentMachineChanged(int machineId)
{
    this->machineId = machineId;

    bool machineHasIsa          = (machine_has_bus(machineId, MACHINE_BUS_ISA) > 0);
    bool machineHasIsaOrSidecar = hasIsaOrSidecarBus(machineId);
    const bool nativeSoftPower = machines[machineId].init == machine_ibm5140_init;

    ui->pushButtonConfigureRTC->setEnabled(machineHasIsaOrSidecar);
    ui->comboBoxRTC->setEnabled(machineHasIsaOrSidecar);
    ui->checkBoxISABugger->setEnabled(machineHasIsa);
    ui->pushButtonConfigureUT->setEnabled(unittester_enabled > 0);
    ui->checkBoxKeyCard->setEnabled(machineHasIsa);
    ui->pushButtonConfigureKeyCard->setEnabled(novell_keycard_enabled > 0);
    ui->checkBoxSoftPower->setEnabled(machineHasIsa && !nativeSoftPower);
    ui->pushButtonConfigureSoftPower->setEnabled(machineHasIsa && !nativeSoftPower && softpower_card_enabled);

    ui->checkBoxISABugger->setChecked((machineHasIsa && (bugger_enabled > 0)) ? true : false);
    ui->checkBoxPOSTCard->setChecked(postcard_enabled > 0 ? true : false);
    ui->checkBoxUnitTester->setChecked(unittester_enabled > 0 ? true : false);
    ui->checkBoxKeyCard->setChecked((machineHasIsa && (novell_keycard_enabled > 0)) ? true : false);
    /* Native power hardware must not overwrite the optional card preference. */
    ui->checkBoxSoftPower->setChecked(nativeSoftPower || (machineHasIsa && softpower_card_enabled));

    scRTC->removeRows();
    ui->comboBoxRTC->clear();

    for (uint8_t i = 0; i < ISAMEM_MAX; ++i) {
        scMemExpCard[i]->removeRows();
        if (auto *cb = findChild<QComboBox *>(QString("comboBoxMemExpCard%1").arg(i + 1)))
            cb->clear();
    }

    for (uint8_t i = 0; i < ISAROM_MAX; ++i) {
        scIsaRomCard[i]->removeRows();
        if (auto *cb = findChild<QComboBox *>(QString("comboBoxIsaRomCard%1").arg(i + 1)))
            cb->clear();
    }

    int selectedRow = 0;

    // ISA RTC Cards
    auto *model = ui->comboBoxRTC->model();
    Models::Batch rtcRows(model);
    for (const auto &rtc : Models::Devices(isartc_get_device, isartc_get_internal_name, nullptr, 0)) {
        if (!device_is_valid(rtc.dev, machineId))
            continue;

            ... (file continues unchanged)
